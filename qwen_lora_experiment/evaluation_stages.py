"""Dependency-injected NLL, PIQA, and pilot MMLU stage execution."""

from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
import torch
from .checkpointing import CheckpointContext, load_best_adapter
from .data import validate_piqa_rows
from .evaluation_contracts import (
    EvaluationBundle,
    EvaluationContext,
    MMLU_EVALUATION_KIND,
    MMLU_SIDECAR_DIRECTORY,
    MmluEvaluationResult,
    NLL_EVALUATION_KIND,
    NllEvaluationResult,
    PILOT_EVALUATION_SCOPE,
    PILOT_NLL_EVALUATION_FILENAME,
    PIQA_EVALUATION_KIND,
    PiqaEvaluationResult,
    _BestLoader,
    _evaluation_artifacts,
    _json_object,
    _mmlu_contract,
)
from .evaluators.common import ModelForward
from .evaluation_predictions import (
    _create_or_validate_immutable_sidecar_record,
    _create_or_validate_immutable_stage_record,
    _encode_prediction_records,
    _load_mmlu_prediction_identity_record_at,
    _load_nll_stage_record,
    _load_prediction_identity_record,
    _load_reusable_mmlu_predictions_at,
    _load_reusable_piqa_predictions,
    _mmlu_prediction_identity,
    _mmlu_prediction_identity_record,
    _mmlu_sidecar_namespace_is_current,
    _normalize_mmlu_data_identity,
    _open_mmlu_sidecar_directory,
    _piqa_prediction_identity,
    _piqa_prediction_identity_record,
    _require_new_output_path,
    _safe_regular_file_bytes,
    _safe_regular_file_bytes_at,
    _write_or_validate_prediction_records,
    _write_or_validate_sidecar_predictions,
)
from .evaluation_records import (
    _experiment_identity,
    _grouped_piqa_reference_alignment_seed,
    _is_grouped_piqa_reference_quality_execution,
    _reject_production_evaluation_test_seams,
    _require_pilot_nll_prerequisite,
    _require_stage_identity_match,
    _stage_record,
)
from .evaluation_scoring import (
    _canonical_mmlu_expected_test_ids,
    _normalize_mmlu_expected_test_ids,
    score_mmlu_rows,
    score_piqa_rows,
    score_wikitext_blocks,
)
from .evaluation_sources import (
    _evaluation_source_kind,
    _load_evaluation_source,
    _require_same_evaluation_source,
)
from .evaluators.common import resolve_device
from .paths import canonical_json, sha256_bytes, sha256_file
from .telemetry import RunStateStore, RunStateTransitionError


def _run_nll_evaluation_stage_impl(
    run_dir: str | Path,
    *,
    model: object,
    kernel_bank: object | None,
    checkpoint_context: EvaluationContext,
    validation_blocks: object,
    test_blocks: object,
    execution: Mapping[str, object],
    data_identity: Mapping[str, object],
    model_identity: Mapping[str, object],
    source_identity: Mapping[str, object],
    experiment_identity: Mapping[str, object],
    evaluation_mode: str = "pilot",
    state_store: RunStateStore | None = None,
    best_loader: _BestLoader | None = load_best_adapter,
    model_forward: ModelForward | None = None,
    device: torch.device | str | None = None,
    sequence_length: int | None = None,
) -> NllEvaluationResult:
    """Publish validation/test NLL only; this stage never reads PIQA rows."""
    _reject_production_evaluation_test_seams(
        checkpoint_context, best_loader=best_loader, model_forward=model_forward
    )
    artifacts = _evaluation_artifacts(evaluation_mode)
    _evaluation_source_kind(checkpoint_context)
    destination, store = _prepare_stage(
        run_dir,
        checkpoint_context=checkpoint_context,
        state_store=state_store,
        best_loader=best_loader,
    )
    final_path = destination / artifacts.nll
    normalized_identity = _experiment_identity(
        experiment_identity, checkpoint_context.method
    )
    try:
        evaluation_source = _load_evaluation_source(
            destination,
            model=model,
            kernel_bank=kernel_bank,
            checkpoint_context=checkpoint_context,
            best_loader=best_loader,
        )
        validation = score_wikitext_blocks(
            model=model,
            blocks=validation_blocks,
            kernel_bank=kernel_bank,
            model_forward=model_forward,
            device=device,
            sequence_length=sequence_length,
        )
        test = score_wikitext_blocks(
            model=model,
            blocks=test_blocks,
            kernel_bank=kernel_bank,
            model_forward=model_forward,
            device=device,
            sequence_length=sequence_length,
        )
        record = _stage_record(
            kind=NLL_EVALUATION_KIND,
            checkpoint_context=checkpoint_context,
            evaluation_source=evaluation_source,
            execution=execution,
            data_identity=data_identity,
            model_identity=model_identity,
            source_identity=source_identity,
            experiment_identity=normalized_identity,
            payload={"validation": validation.as_dict(), "test": test.as_dict()},
            evaluation_scope=PILOT_EVALUATION_SCOPE,
        )
        _create_or_validate_immutable_stage_record(final_path, record, "NLL evaluation")
        return NllEvaluationResult(
            final_path=final_path,
            best_checkpoint=dict(evaluation_source),
            validation=validation,
            test=test,
        )
    except BaseException as exc:
        _mark_evaluation_failed(store, exc)
        raise


def _run_piqa_evaluation_stage_impl(
    run_dir: str | Path,
    *,
    model: object,
    tokenizer: object,
    kernel_bank: object | None,
    checkpoint_context: EvaluationContext,
    piqa_rows: Sequence[Mapping[str, object]],
    execution: Mapping[str, object],
    data_identity: Mapping[str, object],
    model_identity: Mapping[str, object],
    source_identity: Mapping[str, object],
    experiment_identity: Mapping[str, object],
    evaluation_mode: str = "pilot",
    state_store: RunStateStore | None = None,
    best_loader: _BestLoader | None = load_best_adapter,
    model_forward: ModelForward | None = None,
    device: torch.device | str | None = None,
) -> PiqaEvaluationResult:
    """Publish PIQA after this run's matching pilot NLL record is available."""
    _reject_production_evaluation_test_seams(
        checkpoint_context, best_loader=best_loader, model_forward=model_forward
    )
    artifacts = _evaluation_artifacts(evaluation_mode)
    _evaluation_source_kind(checkpoint_context)
    destination, store = _prepare_stage(
        run_dir,
        checkpoint_context=checkpoint_context,
        state_store=state_store,
        best_loader=best_loader,
    )
    nll_path = destination / artifacts.nll
    final_path = destination / artifacts.piqa
    predictions_path = destination / artifacts.piqa_predictions
    predictions_identity_path = destination / artifacts.piqa_predictions_identity
    normalized_identity = _experiment_identity(
        experiment_identity, checkpoint_context.method
    )
    nll_record = _load_nll_stage_record(nll_path)
    _require_stage_identity_match(
        nll_record,
        checkpoint_context=checkpoint_context,
        execution=execution,
        data_identity=data_identity,
        model_identity=model_identity,
        source_identity=source_identity,
        experiment_identity=normalized_identity,
    )
    _require_pilot_nll_prerequisite(nll_record)
    try:
        _require_new_output_path(final_path, "PIQA evaluation")
        evaluation_source = _load_evaluation_source(
            destination,
            model=model,
            kernel_bank=kernel_bank,
            checkpoint_context=checkpoint_context,
            best_loader=best_loader,
        )
        _require_same_evaluation_source(nll_record, evaluation_source=evaluation_source)
        grouped_reference_alignment: dict[str, object] | None = None
        execution_record = _json_object(execution, "execution")
        quality_execution = execution_record.get("grouped_piqa_quality_execution")
        if quality_execution is not None:
            if not _is_grouped_piqa_reference_quality_execution(quality_execution):
                raise ValueError(
                    "Grouped PIQA reference execution evidence is malformed"
                )
            replacement_layers = normalized_identity.get(
                "replaced_layer_ids_zero_based"
            )
            if (
                not isinstance(replacement_layers, list)
                or not replacement_layers
                or any((type(layer_id) is not int for layer_id in replacement_layers))
                or (replacement_layers != sorted(set(replacement_layers)))
            ):
                raise ValueError(
                    "Grouped PIQA reference alignment requires selected replacement layers"
                )
            from .attention.grouped import grouped_quadratic_reference_quality_alignment

            alignment_seed = _grouped_piqa_reference_alignment_seed(checkpoint_context)
            alignment_device = resolve_device(model, device)
            grouped_reference_alignment = {
                "schema": "grouped_piqa_reference_alignment_set_v1",
                "sequence_length": 128,
                "seed": alignment_seed,
                "layers": [
                    grouped_quadratic_reference_quality_alignment(
                        parameter_bank=kernel_bank,
                        layer_id=layer_id,
                        device=alignment_device,
                        seed=alignment_seed,
                    )
                    for layer_id in replacement_layers
                ],
            }
        normalized_rows = validate_piqa_rows(piqa_rows)
        prediction_identity = _piqa_prediction_identity(
            checkpoint_context=checkpoint_context,
            evaluation_source=evaluation_source,
            execution=execution,
            data_identity=data_identity,
            model_identity=model_identity,
            experiment_identity=normalized_identity,
            normalized_rows=normalized_rows,
        )
        reusable = _load_reusable_piqa_predictions(
            predictions_path=predictions_path,
            identity_path=predictions_identity_path,
            expected_identity=prediction_identity,
        )
        if reusable is None:
            piqa = score_piqa_rows(
                model=model,
                tokenizer=tokenizer,
                rows=normalized_rows,
                kernel_bank=kernel_bank,
                model_forward=model_forward,
                device=device,
            )
            encoded_predictions = _encode_prediction_records(piqa.predictions)
            prediction_record = _piqa_prediction_identity_record(
                prediction_identity, encoded_predictions=encoded_predictions
            )
            _create_or_validate_immutable_stage_record(
                predictions_identity_path, prediction_record, "PIQA prediction identity"
            )
            _write_or_validate_prediction_records(
                predictions_path,
                encoded_predictions=encoded_predictions,
                expected_identity=prediction_record,
            )
        else:
            piqa = reusable
        validated_predictions = _load_reusable_piqa_predictions(
            predictions_path=predictions_path,
            identity_path=predictions_identity_path,
            expected_identity=prediction_identity,
        )
        if validated_predictions is None:
            raise ValueError("PIQA predictions disappeared before final publication")
        piqa = validated_predictions
        prediction_identity_record = _load_prediction_identity_record(
            predictions_identity_path
        )
        prediction_sha256 = prediction_identity_record["prediction_sha256"]
        identity_sha256 = sha256_bytes(
            _safe_regular_file_bytes(
                predictions_identity_path, "PIQA prediction identity"
            )
        )
        payload: dict[str, object] = {
            "piqa": piqa.summary(),
            "piqa_prediction_identity": prediction_identity,
            "piqa_predictions": {
                "path": str(predictions_path.resolve()),
                "sha256": prediction_sha256,
                "identity_path": str(predictions_identity_path.resolve()),
                "identity_sha256": identity_sha256,
            },
        }
        if grouped_reference_alignment is not None:
            payload["grouped_piqa_reference_alignment"] = grouped_reference_alignment
        payload["pilot_nll_prerequisite"] = {
            "path": str(nll_path.resolve()),
            "sha256": sha256_file(nll_path),
        }
        _create_or_validate_immutable_stage_record(
            final_path,
            _stage_record(
                kind=PIQA_EVALUATION_KIND,
                checkpoint_context=checkpoint_context,
                evaluation_source=evaluation_source,
                execution=execution,
                data_identity=data_identity,
                model_identity=model_identity,
                source_identity=source_identity,
                experiment_identity=normalized_identity,
                payload=payload,
                evaluation_scope=PILOT_EVALUATION_SCOPE,
            ),
            "PIQA evaluation",
        )
        store.transition(
            "evaluating",
            evaluation_stage="piqa",
            nll_evaluation_path=str(nll_path.resolve()),
            nll_evaluation_sha256=sha256_file(nll_path),
        )
        store.transition(
            "completed",
            final_eval_path=str(final_path.resolve()),
            final_eval_sha256=sha256_file(final_path),
        )
        return PiqaEvaluationResult(
            final_path=final_path,
            predictions_path=predictions_path,
            best_checkpoint=dict(evaluation_source),
            piqa=piqa,
        )
    except BaseException as exc:
        _mark_evaluation_failed(store, exc)
        raise


def run_nll_evaluation_stage(*args: object, **kwargs: object) -> NllEvaluationResult:
    """Dependency-injected NLL evaluator for test checkpoint contexts."""
    context = kwargs.get("checkpoint_context")
    if (
        isinstance(context, CheckpointContext)
        and context.config_identity.get("checkpoint_context_kind") == "formal"
    ):
        raise ValueError("production NLL evaluation requires the private bundle path")
    return _run_nll_evaluation_stage_impl(*args, **kwargs)


def _is_production_pilot_context(value: object) -> bool:
    return (
        isinstance(value, CheckpointContext)
        and value.config_identity.get("checkpoint_context_kind") == "formal"
    )


def _run_pilot_nll_evaluation_stage(
    *args: object, **kwargs: object
) -> NllEvaluationResult:
    """Private production route for one fully-audited pilot checkpoint."""
    context = kwargs.get("checkpoint_context")
    if not _is_production_pilot_context(context):
        raise ValueError(
            "private pilot NLL route requires a production evaluation context"
        )
    return _run_nll_evaluation_stage_impl(*args, evaluation_mode="pilot", **kwargs)


def run_piqa_evaluation_stage(*args: object, **kwargs: object) -> PiqaEvaluationResult:
    """Dependency-injected PIQA evaluator for test checkpoint contexts."""
    context = kwargs.get("checkpoint_context")
    if (
        isinstance(context, CheckpointContext)
        and context.config_identity.get("checkpoint_context_kind") == "formal"
    ):
        raise ValueError("production PIQA evaluation requires the private bundle path")
    return _run_piqa_evaluation_stage_impl(*args, **kwargs)


def _run_pilot_piqa_evaluation_stage(
    *args: object, **kwargs: object
) -> PiqaEvaluationResult:
    """Private production route that binds PIQA to the same pilot NLL record."""
    context = kwargs.get("checkpoint_context")
    if not _is_production_pilot_context(context):
        raise ValueError(
            "private pilot PIQA route requires a production evaluation context"
        )
    return _run_piqa_evaluation_stage_impl(*args, evaluation_mode="pilot", **kwargs)


def run_mmlu_evaluation_stage(*args: object, **kwargs: object) -> MmluEvaluationResult:
    """Public test seam for the pilot-only MMLU sidecar endpoint."""
    if kwargs.get("evaluation_mode", "pilot") != "pilot":
        raise ValueError("MMLU evaluation is pilot-only in v1")
    return _run_pilot_mmlu_evaluation_stage(*args, **kwargs)


def _run_pilot_mmlu_evaluation_stage(
    run_dir: str | Path,
    *,
    model: object,
    tokenizer: object,
    kernel_bank: object | None,
    checkpoint_context: CheckpointContext,
    mmlu_dev_rows: Sequence[Mapping[str, object]],
    mmlu_test_rows: Sequence[Mapping[str, object]],
    expected_test_ids: Mapping[str, Sequence[str]],
    mmlu_data_identity: Mapping[str, object],
    execution: Mapping[str, object],
    data_identity: Mapping[str, object],
    model_identity: Mapping[str, object],
    source_identity: Mapping[str, object],
    experiment_identity: Mapping[str, object],
    best_loader: _BestLoader | None = load_best_adapter,
    model_forward: ModelForward | None = None,
    device: torch.device | str | None = None,
    evaluation_mode: str = "pilot",
) -> MmluEvaluationResult:
    """Publish a read-only MMLU sidecar for one existing LoRA pilot run.

    This deliberately bypasses ``_prepare_stage`` and never accepts a state
    store: run status is not part of the post-hoc MMLU lifecycle.
    """
    mmlu_contract = _mmlu_contract()
    if evaluation_mode != "pilot":
        raise ValueError("MMLU evaluation is pilot-only in v1")
    if not isinstance(checkpoint_context, CheckpointContext):
        raise ValueError(
            "pilot MMLU evaluation accepts only a LoRA best_adapter source"
        )
    if not callable(best_loader):
        raise TypeError("pilot MMLU evaluation requires a callable best_loader")
    destination = Path(run_dir)
    if not destination.is_dir():
        raise FileNotFoundError(
            f"MMLU evaluation run directory does not exist: {destination}"
        )
    normalized_identity = _experiment_identity(
        experiment_identity, checkpoint_context.method
    )
    normalized_mmlu_identity = _normalize_mmlu_data_identity(mmlu_data_identity)
    nll_path = destination / PILOT_NLL_EVALUATION_FILENAME
    nll_record = _load_nll_stage_record(nll_path)
    _require_pilot_nll_prerequisite(nll_record)
    _require_stage_identity_match(
        nll_record,
        checkpoint_context=checkpoint_context,
        execution=execution,
        data_identity=data_identity,
        model_identity=model_identity,
        source_identity=source_identity,
        experiment_identity=normalized_identity,
        endpoint_label="MMLU",
    )
    if "evaluation_source" in nll_record or not isinstance(
        nll_record.get("best_checkpoint"), Mapping
    ):
        raise ValueError(
            "pilot MMLU evaluation requires a pilot NLL best_adapter source"
        )
    artifacts = _evaluation_artifacts("pilot")
    assert (
        artifacts.mmlu
        and artifacts.mmlu_predictions
        and artifacts.mmlu_predictions_identity
    )
    evaluation_source = _load_evaluation_source(
        destination,
        model=model,
        kernel_bank=kernel_bank,
        checkpoint_context=checkpoint_context,
        best_loader=best_loader,
    )
    if nll_record["best_checkpoint"] != _json_object(
        evaluation_source, "best_checkpoint"
    ):
        raise ValueError("pilot MMLU and pilot NLL evaluation sources do not match")
    normalized_dev, normalized_test, observed_subject_counts = (
        mmlu_contract.validate_mmlu_bundle(mmlu_dev_rows, mmlu_test_rows)
    )
    canonical_expected_ids = _canonical_mmlu_expected_test_ids(normalized_test)
    if _normalize_mmlu_expected_test_ids(expected_test_ids) != canonical_expected_ids:
        raise ValueError(
            "pilot MMLU expected_test_ids must match the canonical full bundle"
        )
    prediction_identity = _mmlu_prediction_identity(
        checkpoint_context=checkpoint_context,
        evaluation_source=evaluation_source,
        execution=execution,
        data_identity=data_identity,
        model_identity=model_identity,
        experiment_identity=normalized_identity,
        mmlu_data_identity=normalized_mmlu_identity,
        dev_rows=normalized_dev,
        test_rows=normalized_test,
        expected_test_ids=canonical_expected_ids,
    )
    final_path = destination / MMLU_SIDECAR_DIRECTORY / artifacts.mmlu
    predictions_path = destination / MMLU_SIDECAR_DIRECTORY / artifacts.mmlu_predictions
    predictions_identity_path = (
        destination / MMLU_SIDECAR_DIRECTORY / artifacts.mmlu_predictions_identity
    )
    with _open_mmlu_sidecar_directory(destination) as sidecar_descriptor:
        reusable = _load_reusable_mmlu_predictions_at(
            directory_descriptor=sidecar_descriptor,
            predictions_name=artifacts.mmlu_predictions,
            identity_name=artifacts.mmlu_predictions_identity,
            expected_identity=prediction_identity,
        )
        if reusable is None:
            mmlu = score_mmlu_rows(
                model=model,
                tokenizer=tokenizer,
                dev_rows=normalized_dev,
                test_rows=normalized_test,
                expected_test_ids=canonical_expected_ids,
                kernel_bank=kernel_bank,
                model_forward=model_forward,
                device=device,
            )
            encoded_predictions = _encode_prediction_records(
                mmlu.predictions, label="MMLU"
            )
            prediction_record = _mmlu_prediction_identity_record(
                prediction_identity, encoded_predictions=encoded_predictions
            )
            _create_or_validate_immutable_sidecar_record(
                sidecar_descriptor,
                artifacts.mmlu_predictions_identity,
                prediction_record,
                "MMLU prediction identity",
            )
            _write_or_validate_sidecar_predictions(
                sidecar_descriptor,
                artifacts.mmlu_predictions,
                encoded_predictions=encoded_predictions,
                expected_identity=prediction_record,
                label="MMLU",
            )
        else:
            mmlu = reusable
        validated = _load_reusable_mmlu_predictions_at(
            directory_descriptor=sidecar_descriptor,
            predictions_name=artifacts.mmlu_predictions,
            identity_name=artifacts.mmlu_predictions_identity,
            expected_identity=prediction_identity,
        )
        if validated is None:
            raise ValueError("MMLU predictions disappeared before final publication")
        mmlu = validated
        identity_record = _load_mmlu_prediction_identity_record_at(
            sidecar_descriptor, artifacts.mmlu_predictions_identity
        )
        if not _mmlu_sidecar_namespace_is_current(destination, sidecar_descriptor):
            raise ValueError("MMLU sidecar namespace changed during publication")
        payload = {
            "mmlu_protocol": mmlu_contract.MMLU_PROTOCOL,
            "mmlu_data_identity": normalized_mmlu_identity,
            "mmlu_bundle_identifier": normalized_mmlu_identity["bundle_sha256"],
            "mmlu_subject_counts": observed_subject_counts,
            "post_hoc_read_only": True,
            "mmlu": mmlu.summary(),
            "pilot_nll_prerequisite": {
                "path": str(nll_path.absolute()),
                "sha256": sha256_bytes(
                    _safe_regular_file_bytes(nll_path, "NLL evaluation stage")
                ),
            },
            "mmlu_predictions": {
                "path": str(predictions_path.absolute()),
                "sha256": identity_record["prediction_sha256"],
                "identity_path": str(predictions_identity_path.absolute()),
                "identity_sha256": sha256_bytes(
                    _safe_regular_file_bytes_at(
                        sidecar_descriptor,
                        artifacts.mmlu_predictions_identity,
                        "MMLU prediction identity",
                    )
                ),
            },
        }
        _create_or_validate_immutable_sidecar_record(
            sidecar_descriptor,
            artifacts.mmlu,
            _stage_record(
                kind=MMLU_EVALUATION_KIND,
                checkpoint_context=checkpoint_context,
                evaluation_source=evaluation_source,
                execution=execution,
                data_identity=data_identity,
                model_identity=model_identity,
                source_identity=source_identity,
                experiment_identity=normalized_identity,
                payload=payload,
                evaluation_scope=PILOT_EVALUATION_SCOPE,
            ),
            "pilot MMLU evaluation",
        )
        if not _mmlu_sidecar_namespace_is_current(destination, sidecar_descriptor):
            raise ValueError("MMLU sidecar namespace changed during publication")
    return MmluEvaluationResult(
        final_path=final_path,
        predictions_path=predictions_path,
        best_checkpoint=dict(evaluation_source),
        mmlu=mmlu,
    )


def run_nll_evaluation_stage_from_bundle(
    run_dir: str | Path,
    *,
    bundle_builder: Callable[[], EvaluationBundle],
    validation_blocks: object,
    test_blocks: object,
    execution: Mapping[str, object],
    data_identity: Mapping[str, object],
    model_identity: Mapping[str, object],
    source_identity: Mapping[str, object],
    experiment_identity: Mapping[str, object],
    state_store: RunStateStore | None = None,
    best_loader: _BestLoader | None = load_best_adapter,
    sequence_length: int | None = None,
) -> NllEvaluationResult:
    """Construct a runtime bundle for the NLL stage."""
    if not callable(bundle_builder):
        raise TypeError("bundle_builder must be callable")
    destination = Path(run_dir)
    store = state_store or RunStateStore(destination / "status.json")
    try:
        bundle = bundle_builder()
        if not isinstance(bundle, EvaluationBundle):
            raise TypeError("bundle_builder must return an EvaluationBundle")
    except BaseException as exc:
        _mark_evaluation_failed(store, exc)
        raise
    return run_nll_evaluation_stage(
        destination,
        model=bundle.model,
        kernel_bank=bundle.kernel_bank,
        checkpoint_context=bundle.checkpoint_context,
        validation_blocks=validation_blocks,
        test_blocks=test_blocks,
        execution=execution,
        data_identity=data_identity,
        model_identity=model_identity,
        source_identity=source_identity,
        experiment_identity=experiment_identity,
        state_store=store,
        best_loader=best_loader,
        model_forward=bundle.model_forward,
        device=bundle.device,
        sequence_length=sequence_length,
    )


def _prepare_stage(
    run_dir: str | Path,
    *,
    checkpoint_context: EvaluationContext,
    state_store: RunStateStore | None,
    best_loader: _BestLoader | None,
) -> tuple[Path, RunStateStore]:
    destination = Path(run_dir)
    if not destination.is_dir():
        raise FileNotFoundError(
            f"evaluation run directory does not exist: {destination}"
        )
    if not isinstance(checkpoint_context, CheckpointContext):
        raise TypeError("checkpoint_context is not a supported evaluation context")
    if not callable(best_loader):
        raise TypeError("checkpoint-backed evaluation requires a callable best_loader")
    store = state_store or RunStateStore(destination / "status.json")
    status = store.read()
    if status["state"] != "trained":
        raise RunStateTransitionError(
            f"evaluation stage requires run status trained; found {status['state']!r}"
        )
    return (destination, store)


def _mark_evaluation_failed(store: RunStateStore, error: BaseException) -> None:
    """Best-effort terminal failure evidence without masking the root error."""
    try:
        current = store.read()["state"]
        if current in ("trained", "evaluating"):
            store.transition("failed", failure=error)
    except BaseException:
        pass
