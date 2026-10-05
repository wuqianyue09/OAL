"""Public evaluation API; implementations live in responsibility-specific modules."""

from .evaluation_contracts import (
    BlockNllScore as BlockNllScore,
    CandidateScore as CandidateScore,
    EVALUATION_PROTOCOL_VERSION as EVALUATION_PROTOCOL_VERSION,
    EvaluationBundle as EvaluationBundle,
    EvaluationContext as EvaluationContext,
    LanguageModelScore as LanguageModelScore,
    MMLU_EVALUATION_KIND as MMLU_EVALUATION_KIND,
    MMLU_SIDECAR_DIRECTORY as MMLU_SIDECAR_DIRECTORY,
    MmluEvaluationResult as MmluEvaluationResult,
    MmluScore as MmluScore,
    NLL_EVALUATION_KIND as NLL_EVALUATION_KIND,
    NllEvaluationResult as NllEvaluationResult,
    PILOT_EVALUATION_SCOPE as PILOT_EVALUATION_SCOPE,
    PILOT_MMLU_EVALUATION_FILENAME as PILOT_MMLU_EVALUATION_FILENAME,
    PILOT_MMLU_PREDICTIONS_FILENAME as PILOT_MMLU_PREDICTIONS_FILENAME,
    PILOT_MMLU_PREDICTIONS_IDENTITY_FILENAME as PILOT_MMLU_PREDICTIONS_IDENTITY_FILENAME,
    PILOT_NLL_EVALUATION_FILENAME as PILOT_NLL_EVALUATION_FILENAME,
    PILOT_PIQA_EVALUATION_FILENAME as PILOT_PIQA_EVALUATION_FILENAME,
    PILOT_PIQA_PREDICTIONS_FILENAME as PILOT_PIQA_PREDICTIONS_FILENAME,
    PILOT_PIQA_PREDICTIONS_IDENTITY_FILENAME as PILOT_PIQA_PREDICTIONS_IDENTITY_FILENAME,
    PIQA_EVALUATION_KIND as PIQA_EVALUATION_KIND,
    PiqaEvaluationResult as PiqaEvaluationResult,
    PiqaScore as PiqaScore,
)
from .evaluation_scoring import (
    format_piqa_candidate as format_piqa_candidate,
    format_piqa_prompt as format_piqa_prompt,
    score_mmlu_rows as score_mmlu_rows,
    score_piqa_rows as score_piqa_rows,
    score_wikitext_blocks as score_wikitext_blocks,
)
from .evaluation_sources import _evaluation_source_kind as _evaluation_source_kind
from .evaluation_records import (
    _require_stage_identity_match as _require_stage_identity_match,
)
from .evaluation_predictions import (
    _encode_prediction_records as _encode_prediction_records,
    _publish_prediction_bytes as _publish_prediction_bytes,
    _write_or_validate_prediction_records as _write_or_validate_prediction_records,
)
from .evaluation_stages import (
    _run_nll_evaluation_stage_impl as _run_nll_evaluation_stage_impl,
    _run_pilot_mmlu_evaluation_stage as _run_pilot_mmlu_evaluation_stage,
    _run_pilot_nll_evaluation_stage as _run_pilot_nll_evaluation_stage,
    _run_pilot_piqa_evaluation_stage as _run_pilot_piqa_evaluation_stage,
    _run_piqa_evaluation_stage_impl as _run_piqa_evaluation_stage_impl,
    run_mmlu_evaluation_stage as run_mmlu_evaluation_stage,
    run_nll_evaluation_stage as run_nll_evaluation_stage,
    run_nll_evaluation_stage_from_bundle as run_nll_evaluation_stage_from_bundle,
    run_piqa_evaluation_stage as run_piqa_evaluation_stage,
)
from .checkpointing import load_best_adapter as load_best_adapter
from .paths import sha256_bytes as sha256_bytes
