"""Teacher-forced PIQA, MMLU, and WikiText scoring without model assembly."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
import math
import torch
from torch import Tensor
from torch.nn import functional as F
from .data import validate_piqa_rows
from .evaluators.common import (
    ModelForward,
    evaluation_mode,
    forward_logits,
    resolve_device,
    tokenize_ids,
)
from .paths import sha256_bytes
from .evaluation_contracts import (
    BlockNllScore,
    CandidateScore,
    LanguageModelScore,
    MmluScore,
    PiqaScore,
    _mmlu_contract,
)


def format_piqa_prompt(goal: str) -> str:
    """Format exactly the declared PIQA prompt, stripping only ``goal``."""
    if not isinstance(goal, str):
        raise TypeError("PIQA goal must be a string")
    return f"Question: {goal.strip()}\nAnswer:"


def format_piqa_candidate(answer: str) -> str:
    """Return the declared continuation: one space plus ``answer.strip()``."""
    if not isinstance(answer, str):
        raise TypeError("PIQA answer must be a string")
    return " " + answer.strip()


def score_piqa_rows(
    *,
    model: object,
    tokenizer: object,
    rows: Sequence[Mapping[str, object]],
    kernel_bank: object | None = None,
    model_forward: ModelForward | None = None,
    device: torch.device | str | None = None,
) -> PiqaScore:
    """Score PIQA choices with continuation-only, teacher-forced likelihood.

    Each candidate is independently tokenized as ``prompt + continuation``.
    The separately tokenized prompt must be a *proper* prefix of those full
    IDs; this intentionally rejects tokenizers whose boundary behaviour would
    make a continuation score ambiguous.  Every model call has batch size one,
    no padding or attention mask, and ``use_cache=False``.
    """
    normalized_rows = validate_piqa_rows(rows)
    if not normalized_rows:
        raise ValueError("PIQA evaluation requires at least one row")
    resolved_device = resolve_device(model, device)
    predictions: list[dict[str, object]] = []
    raw_correct = 0
    norm_correct = 0
    with evaluation_mode(model, kernel_bank), torch.inference_mode():
        for row_index, row in enumerate(normalized_rows):
            goal = row["goal"]
            answer_one = row["sol1"]
            answer_two = row["sol2"]
            label = row["label"]
            assert (
                isinstance(goal, str)
                and isinstance(answer_one, str)
                and isinstance(answer_two, str)
            )
            assert type(label) is int
            prompt = format_piqa_prompt(goal)
            prompt_ids = tokenize_ids(
                tokenizer, prompt, label=f"PIQA row {row_index} prompt"
            )
            candidates = (
                format_piqa_candidate(answer_one),
                format_piqa_candidate(answer_two),
            )
            scores = tuple(
                (
                    _score_piqa_candidate(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        prompt_ids=prompt_ids,
                        candidate=candidate,
                        candidate_index=candidate_index,
                        kernel_bank=kernel_bank,
                        model_forward=model_forward,
                        device=resolved_device,
                        row_index=row_index,
                    )
                    for (candidate_index, candidate) in enumerate(candidates)
                )
            )
            prediction = _argmax_first((score.total_log_likelihood for score in scores))
            prediction_norm = _argmax_first(
                (score.mean_log_likelihood for score in scores)
            )
            raw_correct += int(prediction == label)
            norm_correct += int(prediction_norm == label)
            predictions.append(
                {
                    "id": row["id"],
                    "label": label,
                    "prompt": prompt,
                    "prediction": prediction,
                    "prediction_norm": prediction_norm,
                    "raw_correct": prediction == label,
                    "normalized_correct": prediction_norm == label,
                    "candidates": [score.as_dict() for score in scores],
                }
            )
    count = len(predictions)
    return PiqaScore(
        predictions=tuple(predictions),
        raw_accuracy=raw_correct / count,
        acc_norm=norm_correct / count,
    )


def score_mmlu_rows(
    *,
    model: object,
    tokenizer: object,
    dev_rows: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
    expected_test_ids: Mapping[str, Sequence[str]],
    kernel_bank: object | None = None,
    model_forward: ModelForward | None = None,
    device: torch.device | str | None = None,
) -> MmluScore:
    """Score MMLU answer letters with one B=1 next-token forward per row.

    The supplied expected IDs are part of the call contract so that callers
    cannot accidentally report a filtered or incomplete test set.
    """
    mmlu_contract = _mmlu_contract()
    normalized_dev = mmlu_contract.validate_mmlu_rows(dev_rows)
    normalized_test = mmlu_contract.validate_mmlu_rows(test_rows)
    expected_by_subject = _normalize_mmlu_expected_test_ids(expected_test_ids)
    _require_exact_mmlu_test_coverage(normalized_test, expected_by_subject)
    demonstrations = _mmlu_demonstrations_by_subject(
        normalized_dev, expected_by_subject
    )
    resolved_device = resolve_device(model, device)
    predictions: list[dict[str, object]] = []
    correct = 0
    with evaluation_mode(model, kernel_bank), torch.inference_mode():
        for row_index, row in enumerate(normalized_test):
            subject = row["subject"]
            answer = row["answer"]
            identifier = row["id"]
            assert (
                isinstance(subject, str)
                and type(answer) is int
                and isinstance(identifier, str)
            )
            prompt = mmlu_contract.format_mmlu_prompt(
                subject, demonstrations[subject], row
            )
            prompt_ids = tokenize_ids(
                tokenizer, prompt, label=f"MMLU row {row_index} prompt"
            )
            candidates = _score_next_token_choices(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                prompt_ids=prompt_ids,
                candidates=mmlu_contract.mmlu_answer_candidates(),
                kernel_bank=kernel_bank,
                model_forward=model_forward,
                device=resolved_device,
                label=f"MMLU row {row_index}",
            )
            prediction = _argmax_first(
                (candidate.total_log_likelihood for candidate in candidates)
            )
            prediction_letter = mmlu_contract.ANSWER_LETTERS[prediction]
            correct += int(prediction == answer)
            predictions.append(
                {
                    "id": identifier,
                    "subject": subject,
                    "label": answer,
                    "label_letter": mmlu_contract.ANSWER_LETTERS[answer],
                    "prediction": prediction,
                    "prediction_letter": prediction_letter,
                    "correct": prediction == answer,
                    "prompt_token_count": len(prompt_ids),
                    "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                    "candidates": [candidate.as_dict() for candidate in candidates],
                }
            )
    summaries: list[dict[str, object]] = []
    for subject in sorted(expected_by_subject):
        subject_predictions = [
            record for record in predictions if record["subject"] == subject
        ]
        subject_correct = sum(
            (bool(record["correct"]) for record in subject_predictions)
        )
        count = len(subject_predictions)
        summaries.append(
            {
                "subject": subject,
                "question_count": count,
                "correct_count": subject_correct,
                "accuracy": subject_correct / count,
            }
        )
    count = len(predictions)
    return MmluScore(
        predictions=tuple(predictions),
        macro_accuracy=sum((float(summary["accuracy"]) for summary in summaries))
        / len(summaries),
        micro_accuracy=correct / count,
        subject_summaries=tuple(summaries),
    )


def score_wikitext_blocks(
    *,
    model: object,
    blocks: object,
    kernel_bank: object | None = None,
    model_forward: ModelForward | None = None,
    device: torch.device | str | None = None,
    sequence_length: int | None = None,
) -> LanguageModelScore:
    """Score complete ``[blocks, N]`` WikiText blocks using shifted NLL only."""
    token_blocks = _validate_wikitext_blocks(blocks, sequence_length=sequence_length)
    resolved_device = resolve_device(model, device)
    total_nll = 0.0
    token_count = 0
    block_records: list[BlockNllScore] = []
    with evaluation_mode(model, kernel_bank), torch.inference_mode():
        for block_index, block in enumerate(token_blocks):
            input_ids = block.to(device=resolved_device, dtype=torch.long).unsqueeze(0)
            logits = forward_logits(
                model=model,
                input_ids=input_ids,
                kernel_bank=kernel_bank,
                model_forward=model_forward,
                label=f"WikiText block {block_index}",
            )
            block_nll = _shifted_nll(
                logits, input_ids, label=f"WikiText block {block_index}"
            )
            block_token_count = input_ids.shape[1] - 1
            total_nll += block_nll
            token_count += block_token_count
            block_records.append(
                BlockNllScore(
                    block_id=block_index,
                    total_nll=block_nll,
                    token_count=block_token_count,
                )
            )
    if token_count <= 0 or not math.isfinite(total_nll):
        raise ValueError("WikiText evaluation produced no finite next-token NLL")
    mean_nll = total_nll / token_count
    ppl = math.exp(mean_nll)
    if not math.isfinite(ppl):
        raise ValueError("WikiText perplexity is non-finite")
    return LanguageModelScore(
        total_nll=total_nll,
        token_count=token_count,
        mean_nll=mean_nll,
        ppl=ppl,
        block_records=tuple(block_records),
    )


def _normalize_mmlu_expected_test_ids(
    value: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("MMLU expected_test_ids must be a non-empty subject mapping")
    normalized: dict[str, tuple[str, ...]] = {}
    for subject, identifiers in value.items():
        if not isinstance(subject, str) or not subject:
            raise ValueError("MMLU expected_test_ids has an invalid subject")
        if not isinstance(identifiers, Sequence) or isinstance(
            identifiers, (str, bytes)
        ):
            raise ValueError(f"MMLU expected_test_ids.{subject} must be a sequence")
        ids = tuple(identifiers)
        if not ids or any((not isinstance(identifier, str) for identifier in ids)):
            raise ValueError(
                f"MMLU expected_test_ids.{subject} must contain non-empty string IDs"
            )
        if len(set(ids)) != len(ids):
            raise ValueError(f"MMLU expected_test_ids.{subject} contains duplicate IDs")
        if any((identifier.split(":", 1)[0] != subject for identifier in ids)):
            raise ValueError(
                f"MMLU expected_test_ids.{subject} has an ID for another subject"
            )
        normalized[subject] = ids
    return normalized


def _canonical_mmlu_expected_test_ids(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, tuple[str, ...]]:
    """Derive the sole permissible coverage map from a validated full bundle."""
    expected: dict[str, list[str]] = {}
    for row in rows:
        subject = row["subject"]
        identifier = row["id"]
        assert isinstance(subject, str) and isinstance(identifier, str)
        expected.setdefault(subject, []).append(identifier)
    return {subject: tuple(identifiers) for (subject, identifiers) in expected.items()}


def _require_exact_mmlu_test_coverage(
    rows: Sequence[Mapping[str, object]],
    expected_by_subject: Mapping[str, Sequence[str]],
) -> None:
    observed_by_subject: dict[str, set[str]] = {}
    for row in rows:
        subject = row["subject"]
        identifier = row["id"]
        assert isinstance(subject, str) and isinstance(identifier, str)
        observed_by_subject.setdefault(subject, set()).add(identifier)
    expected_sets = {
        subject: set(identifiers)
        for (subject, identifiers) in expected_by_subject.items()
    }
    if observed_by_subject != expected_sets:
        raise ValueError("MMLU test rows do not provide exact expected ID coverage")


def _mmlu_demonstrations_by_subject(
    dev_rows: Sequence[Mapping[str, object]],
    expected_by_subject: Mapping[str, Sequence[str]],
) -> dict[str, tuple[dict[str, object], ...]]:
    by_subject: dict[str, list[dict[str, object]]] = {}
    for row in dev_rows:
        subject = row["subject"]
        assert isinstance(subject, str)
        by_subject.setdefault(subject, []).append(dict(row))
    if set(by_subject) != set(expected_by_subject):
        raise ValueError("MMLU development rows must cover exactly the scored subjects")
    demonstrations: dict[str, tuple[dict[str, object], ...]] = {}
    for subject, rows in by_subject.items():
        if len(rows) != 5:
            raise ValueError(
                f"MMLU development rows for {subject} must contain exactly five examples"
            )
        demonstrations[subject] = tuple(rows)
    return demonstrations


def _score_next_token_choices(
    *,
    model: object,
    tokenizer: object,
    prompt: str,
    prompt_ids: tuple[int, ...],
    candidates: Sequence[str],
    kernel_bank: object | None,
    model_forward: ModelForward | None,
    device: torch.device,
    label: str,
) -> tuple[CandidateScore, ...]:
    """Preflight one-token choices, then score all of them from one forward."""
    if len(candidates) != 4:
        raise ValueError(f"{label} must have exactly four answer candidates")
    continuation_ids: list[int] = []
    for candidate_index, candidate in enumerate(candidates):
        full_ids = tokenize_ids(
            tokenizer, prompt + candidate, label=f"{label} candidate {candidate_index}"
        )
        if (
            len(full_ids) <= len(prompt_ids)
            or full_ids[: len(prompt_ids)] != prompt_ids
        ):
            raise ValueError(
                f"{label} candidate {candidate_index} full token ids must have prompt token ids as a proper prefix"
            )
        continuation = full_ids[len(prompt_ids) :]
        if len(continuation) != 1:
            raise ValueError(
                f"{label} candidates must be distinct single-token continuations"
            )
        continuation_ids.append(continuation[0])
    if len(set(continuation_ids)) != len(continuation_ids):
        raise ValueError(
            f"{label} candidates must be distinct single-token continuations"
        )
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits = forward_logits(
        model=model,
        input_ids=input_ids,
        kernel_bank=kernel_bank,
        model_forward=model_forward,
        label=label,
    )
    next_token_logits = logits[:, -1, :]
    targets = torch.tensor(continuation_ids, dtype=torch.long, device=device)
    _validate_target_range(
        targets, next_token_logits.shape[-1], f"{label} continuation"
    )
    log_probabilities = F.log_softmax(next_token_logits.float(), dim=-1)[0]
    scores: list[CandidateScore] = []
    for candidate_index, (candidate, token_id) in enumerate(
        zip(candidates, continuation_ids)
    ):
        likelihood = float(log_probabilities[token_id].item())
        if not math.isfinite(likelihood):
            raise ValueError(
                f"{label} candidate {candidate_index} likelihood is non-finite"
            )
        scores.append(
            CandidateScore(
                index=candidate_index,
                text=candidate,
                prompt_token_count=len(prompt_ids),
                full_token_count=len(prompt_ids) + 1,
                continuation_token_ids=(token_id,),
                token_count=1,
                total_log_likelihood=likelihood,
                mean_log_likelihood=likelihood,
            )
        )
    return tuple(scores)


def _score_piqa_candidate(
    *,
    model: object,
    tokenizer: object,
    prompt: str,
    prompt_ids: tuple[int, ...],
    candidate: str,
    candidate_index: int,
    kernel_bank: object | None,
    model_forward: ModelForward | None,
    device: torch.device,
    row_index: int,
) -> CandidateScore:
    full_ids = tokenize_ids(
        tokenizer,
        prompt + candidate,
        label=f"PIQA row {row_index} candidate {candidate_index}",
    )
    if len(full_ids) <= len(prompt_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"PIQA row {row_index} candidate {candidate_index} full token ids must have prompt token ids as a proper prefix"
        )
    continuation_ids = full_ids[len(prompt_ids) :]
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    logits = forward_logits(
        model=model,
        input_ids=input_ids,
        kernel_bank=kernel_bank,
        model_forward=model_forward,
        label=f"PIQA row {row_index} candidate {candidate_index}",
    )
    continuation_start = len(prompt_ids) - 1
    prediction_logits = logits[:, continuation_start : input_ids.shape[1] - 1, :]
    continuation = torch.tensor([continuation_ids], dtype=torch.long, device=device)
    _validate_target_range(
        continuation, prediction_logits.shape[-1], "PIQA continuation"
    )
    log_probabilities = F.log_softmax(prediction_logits.float(), dim=-1)
    likelihoods = log_probabilities.gather(-1, continuation.unsqueeze(-1)).squeeze(-1)
    total = float(likelihoods.sum().item())
    if not math.isfinite(total):
        raise ValueError(
            f"PIQA row {row_index} candidate {candidate_index} likelihood is non-finite"
        )
    token_count = len(continuation_ids)
    return CandidateScore(
        index=candidate_index,
        text=candidate,
        prompt_token_count=len(prompt_ids),
        full_token_count=len(full_ids),
        continuation_token_ids=continuation_ids,
        token_count=token_count,
        total_log_likelihood=total,
        mean_log_likelihood=total / token_count,
    )


def _shifted_nll(logits: Tensor, input_ids: Tensor, *, label: str) -> float:
    if input_ids.shape[1] < 2:
        raise ValueError(f"{label} requires at least two tokens")
    shifted_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    labels = input_ids[:, 1:].reshape(-1)
    _validate_target_range(labels, logits.shape[-1], f"{label} token ids")
    nll = F.cross_entropy(shifted_logits.float(), labels, reduction="sum")
    result = float(nll.item())
    if not math.isfinite(result):
        raise ValueError(f"{label} NLL is non-finite")
    return result


def _validate_target_range(targets: Tensor, vocabulary_size: int, label: str) -> None:
    if targets.numel() == 0:
        raise ValueError(f"{label} must contain at least one token")
    if targets.min().item() < 0 or targets.max().item() >= vocabulary_size:
        raise ValueError(
            f"{label} is outside model vocabulary range [0, {vocabulary_size})"
        )


def _validate_wikitext_blocks(blocks: object, *, sequence_length: int | None) -> Tensor:
    try:
        token_blocks = torch.as_tensor(blocks)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError("WikiText blocks must be convertible to a tensor") from exc
    if token_blocks.ndim != 2:
        raise ValueError("WikiText blocks must be two-dimensional [blocks, N]")
    if token_blocks.shape[0] <= 0:
        raise ValueError("WikiText blocks must contain at least one complete block")
    if token_blocks.shape[1] < 2:
        raise ValueError("WikiText blocks must contain at least two tokens per block")
    if (
        token_blocks.dtype is torch.bool
        or token_blocks.is_floating_point()
        or token_blocks.is_complex()
    ):
        raise TypeError("WikiText blocks must have an integer dtype")
    if sequence_length is not None:
        if type(sequence_length) is not int or sequence_length < 2:
            raise ValueError("sequence_length must be an integer of at least two")
        if token_blocks.shape[1] != sequence_length:
            raise ValueError(
                "WikiText block length does not match the configured sequence_length"
            )
    return token_blocks


def _argmax_first(values: object) -> int:
    iterator = iter(values)
    try:
        best_value = next(iterator)
    except StopIteration as exc:
        raise ValueError("cannot choose a PIQA prediction from no candidates") from exc
    if not isinstance(best_value, float) or not math.isfinite(best_value):
        raise ValueError("PIQA likelihood must be finite")
    best_index = 0
    for index, value in enumerate(iterator, start=1):
        if not isinstance(value, float) or not math.isfinite(value):
            raise ValueError("PIQA likelihood must be finite")
        if value > best_value:
            best_index = index
            best_value = value
    return best_index
