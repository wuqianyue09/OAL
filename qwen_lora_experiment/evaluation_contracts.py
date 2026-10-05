"""Evaluation score types, source contexts, and artifact naming contracts."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Protocol
import torch
from .checkpointing import CheckpointContext, LoadedCheckpoint
from .evaluators.common import ModelForward
from .paths import canonical_json

PILOT_NLL_EVALUATION_FILENAME = "pilot_nll_eval.json"
PILOT_PIQA_EVALUATION_FILENAME = "pilot_piqa_eval.json"
PILOT_PIQA_PREDICTIONS_FILENAME = "pilot_piqa_validation_predictions.jsonl"
PILOT_PIQA_PREDICTIONS_IDENTITY_FILENAME = (
    "pilot_piqa_validation_predictions_identity.json"
)
MMLU_SIDECAR_DIRECTORY = "mmlu_sidecar"
PILOT_MMLU_EVALUATION_FILENAME = "pilot_mmlu_eval.json"
PILOT_MMLU_PREDICTIONS_FILENAME = "mmlu_test_predictions.jsonl"
PILOT_MMLU_PREDICTIONS_IDENTITY_FILENAME = "mmlu_test_predictions_identity.json"
NLL_EVALUATION_KIND = "qwen_lora_nll_evaluation"
PIQA_EVALUATION_KIND = "qwen_lora_piqa_evaluation"
MMLU_EVALUATION_KIND = "qwen_lora_mmlu_evaluation"
EVALUATION_PROTOCOL_VERSION = "nll-evaluation-v1"
PILOT_EVALUATION_SCOPE = "pilot_single_seed"


@dataclass(frozen=True)
class _EvaluationArtifactNames:
    """The immutable output names associated with one evaluation mode."""

    nll: str
    piqa: str
    piqa_predictions: str
    piqa_predictions_identity: str
    mmlu: str | None = None
    mmlu_predictions: str | None = None
    mmlu_predictions_identity: str | None = None


_PILOT_EVALUATION_ARTIFACTS = _EvaluationArtifactNames(
    nll=PILOT_NLL_EVALUATION_FILENAME,
    piqa=PILOT_PIQA_EVALUATION_FILENAME,
    piqa_predictions=PILOT_PIQA_PREDICTIONS_FILENAME,
    piqa_predictions_identity=PILOT_PIQA_PREDICTIONS_IDENTITY_FILENAME,
    mmlu=PILOT_MMLU_EVALUATION_FILENAME,
    mmlu_predictions=PILOT_MMLU_PREDICTIONS_FILENAME,
    mmlu_predictions_identity=PILOT_MMLU_PREDICTIONS_IDENTITY_FILENAME,
)


class _BestLoader(Protocol):

    def __call__(
        self,
        run_dir: str | Path,
        *,
        model: object,
        kernel_bank: object,
        context: object,
    ) -> LoadedCheckpoint: ...


EvaluationContext = CheckpointContext


@dataclass(frozen=True)
class CandidateScore:
    """One PIQA continuation's teacher-forced likelihood evidence."""

    index: int
    text: str
    prompt_token_count: int
    full_token_count: int
    continuation_token_ids: tuple[int, ...]
    token_count: int
    total_log_likelihood: float
    mean_log_likelihood: float

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "text": self.text,
            "prompt_token_count": self.prompt_token_count,
            "full_token_count": self.full_token_count,
            "continuation_token_ids": list(self.continuation_token_ids),
            "token_count": self.token_count,
            "total_log_likelihood": self.total_log_likelihood,
            "mean_log_likelihood": self.mean_log_likelihood,
        }


@dataclass(frozen=True)
class PiqaScore:
    """Aggregate PIQA results and the complete per-question audit trail."""

    predictions: tuple[dict[str, object], ...]
    raw_accuracy: float
    acc_norm: float

    @property
    def question_count(self) -> int:
        return len(self.predictions)

    def summary(self) -> dict[str, object]:
        return {
            "question_count": self.question_count,
            "raw_accuracy": self.raw_accuracy,
            "acc_norm": self.acc_norm,
        }


@dataclass(frozen=True)
class MmluScore:
    """Aggregate exact-letter MMLU evidence, including every subject result."""

    predictions: tuple[dict[str, object], ...]
    macro_accuracy: float
    micro_accuracy: float
    subject_summaries: tuple[dict[str, object], ...]

    @property
    def question_count(self) -> int:
        return len(self.predictions)

    def summary(self) -> dict[str, object]:
        return {
            "question_count": self.question_count,
            "macro_accuracy": self.macro_accuracy,
            "micro_accuracy": self.micro_accuracy,
            "subject_summaries": [dict(summary) for summary in self.subject_summaries],
        }


@dataclass(frozen=True)
class BlockNllScore:
    """JSON-safe exact NLL evidence for one immutable evaluation block."""

    block_id: int
    total_nll: float
    token_count: int

    def __post_init__(self) -> None:
        if type(self.block_id) is not int or self.block_id < 0:
            raise ValueError("block_id must be a non-negative integer")
        if (
            isinstance(self.total_nll, bool)
            or not isinstance(self.total_nll, (int, float))
            or (not math.isfinite(float(self.total_nll)))
            or (float(self.total_nll) < 0.0)
        ):
            raise ValueError("total_nll must be finite and non-negative")
        if type(self.token_count) is not int or self.token_count <= 0:
            raise ValueError("token_count must be a positive integer")

    def as_dict(self) -> dict[str, object]:
        return {
            "block_id": self.block_id,
            "total_nll": self.total_nll,
            "token_count": self.token_count,
        }


@dataclass(frozen=True)
class LanguageModelScore:
    """The next-token evidence and perplexity for one full-block split."""

    total_nll: float
    token_count: int
    mean_nll: float
    ppl: float
    block_records: tuple[BlockNllScore, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "total_nll": self.total_nll,
            "token_count": self.token_count,
            "mean_nll": self.mean_nll,
            "ppl": self.ppl,
            "blocks": [record.as_dict() for record in self.block_records],
        }


@dataclass(frozen=True)
class EvaluationBundle:
    """Runtime objects supplied by a future real-model setup layer."""

    model: object
    tokenizer: object
    kernel_bank: object | None
    checkpoint_context: EvaluationContext
    device: torch.device | str | None = None
    model_forward: ModelForward | None = None


@dataclass(frozen=True)
class NllEvaluationResult:
    """The irreversible validation/test NLL stage, before PIQA is unlocked."""

    final_path: Path
    best_checkpoint: dict[str, object]
    validation: LanguageModelScore
    test: LanguageModelScore


@dataclass(frozen=True)
class PiqaEvaluationResult:
    """The separately published PIQA stage after this run's NLL completes."""

    final_path: Path
    predictions_path: Path
    best_checkpoint: dict[str, object]
    piqa: PiqaScore


@dataclass(frozen=True)
class MmluEvaluationResult:
    """Create-only post-hoc MMLU sidecar paths and the audited score."""

    final_path: Path
    predictions_path: Path
    best_checkpoint: dict[str, object]
    mmlu: MmluScore


def _json_object(value: Mapping[str, object], label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    try:
        normalized = json.loads(canonical_json(dict(value)))
    except ValueError as exc:
        raise ValueError(f"{label} must be canonical JSON-safe data") from exc
    if not isinstance(normalized, dict):
        raise ValueError(f"{label} must serialize to a JSON object")
    return normalized


def _mmlu_contract() -> ModuleType:
    """Load MMLU-only helpers only after the MMLU endpoint is selected.

    NLL and PIQA remain usable when the optional MMLU data module is absent.
    The deliberately untyped return keeps this import boundary free of an
    import-time dependency on MMLU implementation details.
    """
    from . import mmlu

    return mmlu


def _evaluation_artifacts(evaluation_mode: str) -> _EvaluationArtifactNames:
    if evaluation_mode != "pilot":
        raise ValueError("evaluation_mode must be 'pilot'")
    return _PILOT_EVALUATION_ARTIFACTS
