"""Dependency-light GSM8K text, sampling, and answer contracts."""

from __future__ import annotations
from collections.abc import Callable, Iterable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import random
import re

GSM8K_DATASET_NAME = "openai/gsm8k"
GSM8K_CONFIG_NAME = "main"
GSM8K_TRAIN_COUNT = 7473
GSM8K_TEST_COUNT = 1319
GSM8K_FEWSHOT_COUNT = 5
GSM8K_FEWSHOT_SEED = 1234
GSM8K_PROTOCOL = "gsm8k-harness-5shot-greedy-v1"
GSM8K_BUNDLE_SCHEMA = "qwen_lora_gsm8k_bundle_v1"
GSM8K_BUNDLE_DIRECTORY = "openai_gsm8k_5shot_v1"
GSM8K_FEWSHOT_FILENAME = "fewshot.jsonl"
GSM8K_TEST_FILENAME = "test.jsonl"
GSM8K_MANIFEST_FILENAME = "manifest.json"
_NUMBER_PATTERN = "-?\\s*\\$?\\s*(?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?"
_STRICT_ANSWER = re.compile(f"####\\s*({_NUMBER_PATTERN})")
_FLEXIBLE_ANSWER = re.compile(_NUMBER_PATTERN)
Gsm8kSourceLoader = Callable[[str, str | None], Iterable[Mapping[str, object]]]


def normalize_numeric_answer(value: str) -> str:
    """Return one canonical decimal string without arithmetic correction."""
    if not isinstance(value, str):
        raise TypeError("GSM8K numeric answer must be a string")
    compact = value.replace("$", "").replace(",", "").replace(" ", "").strip()
    if not compact:
        raise ValueError("GSM8K numeric answer must not be empty")
    try:
        number = Decimal(compact)
    except InvalidOperation as exc:
        raise ValueError(f"invalid GSM8K numeric answer: {value!r}") from exc
    if not number.is_finite():
        raise ValueError("GSM8K numeric answer must be finite")
    if number == 0:
        return "0"
    rendered = format(number, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered[1:] if rendered.startswith("+") else rendered


def extract_strict_answer(text: str) -> str | None:
    """Extract the last standard ``#### number`` answer."""
    if not isinstance(text, str):
        raise TypeError("GSM8K answer text must be a string")
    matches = _STRICT_ANSWER.findall(text)
    return None if not matches else normalize_numeric_answer(matches[-1])


def extract_flexible_answer(text: str) -> str | None:
    """Prefer strict extraction, then fall back to the last numeric span."""
    strict = extract_strict_answer(text)
    if strict is not None:
        return strict
    matches = _FLEXIBLE_ANSWER.findall(text)
    return None if not matches else normalize_numeric_answer(matches[-1])


def normalize_gsm8k_train_rows(
    rows: Iterable[Mapping[str, object]], *, expected_count: int = GSM8K_TRAIN_COUNT
) -> list[dict[str, object]]:
    return _normalize_source_rows(rows, split="train", expected_count=expected_count)


def normalize_gsm8k_test_rows(
    rows: Iterable[Mapping[str, object]], *, expected_count: int = GSM8K_TEST_COUNT
) -> list[dict[str, object]]:
    normalized = _normalize_source_rows(
        rows, split="test", expected_count=expected_count
    )
    for row_index, row in enumerate(normalized):
        answer = row["answer"]
        assert isinstance(answer, str)
        gold = extract_strict_answer(answer)
        if gold is None:
            raise ValueError(f"GSM8K test row {row_index} lacks a strict final answer")
        row["gold_answer"] = gold
    return normalized


def build_fewshot_assignments(
    train_rows: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
    *,
    seed: int = GSM8K_FEWSHOT_SEED,
) -> list[dict[str, object]]:
    """Materialize harness-compatible sequential random samples per test row."""
    if type(seed) is not int:
        raise TypeError("GSM8K few-shot seed must be an integer")
    train = _validate_prepared_train_rows(train_rows)
    test = _validate_prepared_test_rows(test_rows)
    if len(train) < GSM8K_FEWSHOT_COUNT:
        raise ValueError("GSM8K few-shot pool must contain at least five rows")
    rng = random.Random(seed)
    assignments: list[dict[str, object]] = []
    for test_index, test_row in enumerate(test):
        indices = rng.sample(range(len(train)), GSM8K_FEWSHOT_COUNT)
        assignments.append(
            {
                "id": test_row["id"],
                "test_index": test_index,
                "demonstration_indices": indices,
                "demonstrations": [dict(train[index]) for index in indices],
            }
        )
    return assignments


def format_gsm8k_prompt(
    demonstrations: Sequence[Mapping[str, object]], question: Mapping[str, object]
) -> str:
    """Render the locked 5-shot harness prompt without a chat template."""
    if len(demonstrations) != GSM8K_FEWSHOT_COUNT:
        raise ValueError("GSM8K prompt requires exactly five demonstrations")
    parts: list[str] = []
    for index, row in enumerate(demonstrations):
        question_text, answer_text = _question_answer(row, f"demonstration {index}")
        parts.append(f"Question: {question_text}\nAnswer: {answer_text}")
    if not isinstance(question, Mapping):
        raise TypeError("GSM8K target question must be a mapping")
    question_text = question.get("question")
    if not isinstance(question_text, str) or not question_text:
        raise ValueError("GSM8K target question text must be non-empty")
    parts.append(f"Question: {question_text}\nAnswer:")
    return "\n\n".join(parts)


def gsm8k_bundle_path(data_root: str | Path) -> Path:
    from .gsm8k_bundle import gsm8k_bundle_path as implementation

    return implementation(data_root)


def load_gsm8k_source(
    split: str, requested_source_revision: str | None
) -> Iterable[Mapping[str, object]]:
    from .gsm8k_bundle import load_gsm8k_source as implementation

    return implementation(split, requested_source_revision)


def installed_datasets_version() -> str:
    from .gsm8k_bundle import installed_datasets_version as implementation

    return implementation()


def prepare_gsm8k_bundle(
    data_root: str | Path,
    *,
    source_loader: Gsm8kSourceLoader | None,
    tokenizer_identity: Mapping[str, object],
    datasets_version: str | None = None,
    requested_source_revision: str | None = None,
    verify_only: bool = False,
    expected_train_count: int = GSM8K_TRAIN_COUNT,
    expected_test_count: int = GSM8K_TEST_COUNT,
) -> dict[str, object]:
    from .gsm8k_bundle import prepare_gsm8k_bundle as implementation

    return implementation(
        data_root,
        source_loader=source_loader,
        tokenizer_identity=tokenizer_identity,
        datasets_version=datasets_version,
        requested_source_revision=requested_source_revision,
        verify_only=verify_only,
        expected_train_count=expected_train_count,
        expected_test_count=expected_test_count,
    )


def validate_gsm8k_bundle_directory(
    bundle_path: str | Path,
    *,
    tokenizer_identity: Mapping[str, object] | None = None,
    datasets_version: str | None = None,
    requested_source_revision: str | None = None,
    expected_train_count: int = GSM8K_TRAIN_COUNT,
    expected_test_count: int = GSM8K_TEST_COUNT,
) -> dict[str, object]:
    from .gsm8k_bundle import validate_gsm8k_bundle_directory as implementation

    return implementation(
        bundle_path,
        tokenizer_identity=tokenizer_identity,
        datasets_version=datasets_version,
        requested_source_revision=requested_source_revision,
        expected_train_count=expected_train_count,
        expected_test_count=expected_test_count,
    )


def load_gsm8k_bundle(
    bundle_path: str | Path,
    *,
    expected_train_count: int = GSM8K_TRAIN_COUNT,
    expected_test_count: int = GSM8K_TEST_COUNT,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    from .gsm8k_bundle import load_gsm8k_bundle as implementation

    return implementation(
        bundle_path,
        expected_train_count=expected_train_count,
        expected_test_count=expected_test_count,
    )


def _normalize_source_rows(
    rows: Iterable[Mapping[str, object]], *, split: str, expected_count: int
) -> list[dict[str, object]]:
    if type(expected_count) is not int or expected_count <= 0:
        raise ValueError("expected_count must be a positive integer")
    normalized: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"GSM8K {split} row {row_index} must be a mapping")
        question, answer = _question_answer(row, f"{split} row {row_index}")
        normalized.append(
            {"id": f"{split}:{row_index}", "question": question, "answer": answer}
        )
    if len(normalized) != expected_count:
        raise ValueError(
            f"GSM8K {split} split requires exactly {expected_count} rows; observed {len(normalized)}"
        )
    return normalized


def _question_answer(row: Mapping[str, object], label: str) -> tuple[str, str]:
    if not isinstance(row, Mapping):
        raise TypeError(f"GSM8K {label} must be a mapping")
    question = row.get("question")
    answer = row.get("answer")
    if not isinstance(question, str) or not question:
        raise ValueError(f"GSM8K {label} question must be a non-empty string")
    if not isinstance(answer, str) or not answer:
        raise ValueError(f"GSM8K {label} answer must be a non-empty string")
    return (question, answer)


def _validate_prepared_train_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        value = _json_object(row, f"GSM8K prepared train row {index}")
        if (
            set(value) != {"id", "question", "answer"}
            or value["id"] != f"train:{index}"
        ):
            raise ValueError(f"GSM8K prepared train row {index} is not canonical")
        _question_answer(value, f"prepared train row {index}")
        normalized.append(value)
    return normalized


def _validate_prepared_test_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        value = _json_object(row, f"GSM8K prepared test row {index}")
        if set(value) != {"id", "question", "answer", "gold_answer"}:
            raise ValueError(f"GSM8K prepared test row {index} fields are invalid")
        if value["id"] != f"test:{index}":
            raise ValueError(f"GSM8K prepared test row {index} ID is not canonical")
        _question_answer(value, f"prepared test row {index}")
        answer = value["answer"]
        assert isinstance(answer, str)
        if value["gold_answer"] != extract_strict_answer(answer):
            raise ValueError(f"GSM8K prepared test row {index} gold answer is invalid")
        normalized.append(value)
    return normalized


def _json_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    try:
        copied = json.loads(
            json.dumps(
                dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON-serializable") from exc
    if not isinstance(copied, dict):
        raise AssertionError("canonical JSON unexpectedly changed object type")
    return copied
