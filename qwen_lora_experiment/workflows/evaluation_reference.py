"""Grouped reference execution sessions and PIQA quality identities."""

from __future__ import annotations
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from ..config import PilotConfig
from ..workflows.errors import OrchestrationError

GROUPED_MMLU_REFERENCE_MAX_SEQUENCE_LENGTH = 4096
GROUPED_PIQA_REFERENCE_MAX_SEQUENCE_LENGTH = 256


def _grouped_piqa_candidate_lengths(tokenizer: object, rows: object) -> list[int]:
    """Return the real full-token lengths that the PIQA scorer will execute."""
    from ..data import validate_piqa_rows
    from ..evaluation_scoring import format_piqa_candidate, format_piqa_prompt
    from ..evaluators.common import tokenize_ids

    if not callable(tokenizer):
        raise OrchestrationError("PIQA tokenizer is unavailable for grouped admission")
    try:
        normalized_rows = validate_piqa_rows(rows)
    except (TypeError, ValueError) as exc:
        raise OrchestrationError("PIQA rows are invalid for grouped admission") from exc
    lengths: list[int] = []
    for row_index, row in enumerate(normalized_rows):
        goal = row["goal"]
        for candidate_index, answer in enumerate((row["sol1"], row["sol2"])):
            try:
                token_ids = tokenize_ids(
                    tokenizer,
                    format_piqa_prompt(goal) + format_piqa_candidate(answer),
                    label=f"PIQA candidate {row_index}:{candidate_index}",
                )
                if len(token_ids) < 2:
                    raise ValueError("PIQA candidate requires at least two tokens")
            except (TypeError, ValueError) as exc:
                raise OrchestrationError(
                    f"PIQA candidate {row_index}:{candidate_index} has no valid runtime token length"
                ) from exc
            lengths.append(len(token_ids))
    return lengths


def _require_grouped_piqa_reference_length_bound(
    candidate_lengths: Sequence[int],
) -> tuple[int, ...]:
    """Keep reference-only PIQA bounded without manufacturing Triton records."""
    if isinstance(candidate_lengths, (str, bytes)):
        raise TypeError("grouped PIQA candidate lengths must be a sequence of integers")
    normalized = tuple(sorted(set(candidate_lengths)))
    if not normalized or any(
        (type(length) is not int or length < 2 for length in normalized)
    ):
        raise OrchestrationError("grouped PIQA candidate lengths are invalid")
    if normalized[-1] > GROUPED_PIQA_REFERENCE_MAX_SEQUENCE_LENGTH:
        raise OrchestrationError(
            f"grouped PIQA reference quality evaluation exceeds the maximum supported logical length {GROUPED_PIQA_REFERENCE_MAX_SEQUENCE_LENGTH}"
        )
    return normalized


def _grouped_piqa_reference_quality_execution(
    execution: Mapping[str, object],
) -> dict[str, object]:
    """Declare the active PIQA reference call without relabelling trained Triton evidence."""
    normalized = dict(execution)
    attention = normalized.get("attention")
    if (
        not isinstance(attention, Mapping)
        or attention.get("method") != "grouped_quadratic"
    ):
        raise OrchestrationError(
            "grouped PIQA reference quality requires grouped OAL training evidence"
        )
    if attention.get("execution") not in {"triton", "hd_block_gemm_causal"}:
        raise OrchestrationError(
            "grouped PIQA reference quality requires supported OAL training attention evidence"
        )
    normalized["grouped_piqa_quality_execution"] = {
        "schema": "grouped_piqa_reference_quality_v1",
        "execution": "reference",
        "public_callable": "oal_attention.oal_attention",
        "scope": "quality_only_no_triton_capability_or_performance_claim",
        "max_sequence_length": GROUPED_PIQA_REFERENCE_MAX_SEQUENCE_LENGTH,
    }
    return normalized


def _piqa_reference_quality_execution(
    config: PilotConfig, execution: Mapping[str, object]
) -> dict[str, object]:
    return _grouped_piqa_reference_quality_execution(execution)


@contextmanager
def _grouped_quadratic_reference_session(
    model: object, *, expected_layer_ids: Sequence[int]
) -> Iterator[None]:
    """Load the Torch-owning temporary reference switch only at runtime."""
    from ..attention.grouped import grouped_quadratic_reference_session

    with grouped_quadratic_reference_session(
        model, expected_layer_ids=expected_layer_ids
    ):
        yield


def _grouped_base_reference_session(
    config: PilotConfig, model: object, *, expected_layer_ids: Sequence[int]
) -> object:
    return _grouped_quadratic_reference_session(
        model, expected_layer_ids=expected_layer_ids
    )


def _load_grouped_piqa_inputs(
    config: PilotConfig, *, run_dir: Path
) -> tuple[object, tuple[dict[str, object], ...]]:
    """Validate PIQA rows and tokenizer against this run's frozen manifest."""
    from .. import run_artifacts
    from .. import paths as file_paths
    from .prepare import (
        _assert_local_tokenizer_files_match_manifest,
        _load_local_pilot_tokenizer,
    )
    from ..data import PIQA_FILENAME, load_piqa_rows, piqa_asset_path

    stored_manifest = run_artifacts._safe_run_artifact_json(
        run_dir, run_artifacts.DATA_MANIFEST_COPY_FILENAME, "data-manifest evidence"
    )
    files = stored_manifest.get("files")
    piqa = stored_manifest.get("piqa")
    if not isinstance(files, Mapping) or not isinstance(piqa, Mapping):
        raise OrchestrationError("PIQA input has no immutable manifest record")
    file_record = files.get(PIQA_FILENAME)
    if (
        not isinstance(file_record, Mapping)
        or piqa.get("filename") != PIQA_FILENAME
        or type((piqa_row_count := piqa.get("row_count"))) is not int
        or (piqa_row_count < 1)
        or (not isinstance(file_record.get("sha256"), str))
        or (type(file_record.get("bytes")) is not int)
    ):
        raise OrchestrationError("PIQA input manifest record is malformed")
    path = piqa_asset_path(config.data_root)
    try:
        matches_manifest = (
            file_paths.sha256_file(path) == file_record["sha256"]
            and path.stat().st_size == file_record["bytes"]
        )
    except OSError as exc:
        raise OrchestrationError("PIQA input is unavailable") from exc
    if not matches_manifest:
        raise OrchestrationError(
            "PIQA input does not match immutable manifest evidence"
        )
    try:
        rows = tuple(load_piqa_rows(path))
    except (OSError, TypeError, ValueError) as exc:
        raise OrchestrationError("PIQA input cannot be parsed canonically") from exc
    if len(rows) != piqa_row_count:
        raise OrchestrationError(
            "PIQA input row count does not match immutable evidence"
        )
    try:
        _assert_local_tokenizer_files_match_manifest(config, stored_manifest)
        tokenizer = _load_local_pilot_tokenizer(Path(config.model_path))
    except Exception as exc:
        raise OrchestrationError(
            "PIQA tokenizer does not match immutable input evidence"
        ) from exc
    return (tokenizer, rows)
