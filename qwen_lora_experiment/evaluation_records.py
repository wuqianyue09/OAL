"""Evaluation record identities and same-run NLL prerequisites."""

from __future__ import annotations
from collections.abc import Mapping
from .checkpointing import CheckpointContext, load_best_adapter
from .config import GROUPED_BASE_METHODS
from .evaluation_contracts import (
    EVALUATION_PROTOCOL_VERSION,
    EvaluationContext,
    PILOT_EVALUATION_SCOPE,
    _BestLoader,
    _json_object,
)
from .evaluators.common import ModelForward
from .paths import SCHEMA_VERSION
from .operator_imports import normalize_operator_identity


def _experiment_identity(value: Mapping[str, object], method: str) -> dict[str, object]:
    identity = _json_object(value, "experiment_identity")
    version = identity.get("experiment_protocol_version")
    if not isinstance(version, str) or not version:
        raise ValueError(
            "experiment_identity.experiment_protocol_version must be a non-empty string"
        )
    replacement = identity.get("replaced_layer_ids_zero_based")
    if not isinstance(replacement, list):
        raise ValueError(
            "experiment_identity.replaced_layer_ids_zero_based must be an array of layer IDs"
        )
    if any((type(layer_id) is not int or layer_id < 0 for layer_id in replacement)):
        raise ValueError(
            "experiment_identity.replaced_layer_ids_zero_based must contain non-negative integers"
        )
    if replacement != sorted(set(replacement)):
        raise ValueError(
            "experiment_identity.replaced_layer_ids_zero_based must be sorted with no duplicates"
        )
    declared_method = identity.get("method")
    if declared_method is not None and declared_method != method:
        raise ValueError(
            "experiment_identity.method must match checkpoint_context.method"
        )
    identity["method"] = method
    return identity


def _stage_record(
    *,
    kind: str,
    checkpoint_context: EvaluationContext,
    evaluation_source: Mapping[str, object],
    execution: Mapping[str, object],
    data_identity: Mapping[str, object],
    model_identity: Mapping[str, object],
    source_identity: Mapping[str, object],
    experiment_identity: Mapping[str, object],
    payload: Mapping[str, object],
    evaluation_scope: str | None = None,
) -> dict[str, object]:
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "method": checkpoint_context.method,
        "execution": _json_object(execution, "execution"),
        "data_identity": _json_object(data_identity, "data_identity"),
        "model_identity": _json_object(model_identity, "model_identity"),
        "source_identity": _json_object(source_identity, "source_identity"),
        "experiment_identity": _json_object(experiment_identity, "experiment_identity"),
        "checkpoint_context": {
            "config_identity": _json_object(
                checkpoint_context.config_identity, "checkpoint_context.config_identity"
            ),
            "data_identity": _json_object(
                checkpoint_context.data_identity, "checkpoint_context.data_identity"
            ),
            "model_identity": _json_object(
                checkpoint_context.model_identity, "checkpoint_context.model_identity"
            ),
        },
        **_json_object(payload, "stage payload"),
    }
    normalized_source = _json_object(evaluation_source, "evaluation_source")
    record["best_checkpoint"] = normalized_source
    if evaluation_scope is not None:
        if evaluation_scope != PILOT_EVALUATION_SCOPE:
            raise ValueError("evaluation_scope is not the pilot publication scope")
        record["evaluation_scope"] = evaluation_scope
    return record


def _require_stage_identity_match(
    nll_record: Mapping[str, object],
    *,
    checkpoint_context: EvaluationContext,
    execution: Mapping[str, object],
    data_identity: Mapping[str, object],
    model_identity: Mapping[str, object],
    source_identity: Mapping[str, object],
    experiment_identity: Mapping[str, object],
    endpoint_label: str = "PIQA",
) -> None:
    if not isinstance(endpoint_label, str) or not endpoint_label:
        raise TypeError("endpoint_label must be a non-empty string")
    nll_record = normalize_operator_identity(nll_record)
    expected_context = normalize_operator_identity(
        {
            "config_identity": _json_object(
                checkpoint_context.config_identity, "checkpoint_context.config_identity"
            ),
            "data_identity": _json_object(
                checkpoint_context.data_identity, "checkpoint_context.data_identity"
            ),
            "model_identity": _json_object(
                checkpoint_context.model_identity, "checkpoint_context.model_identity"
            ),
        }
    )
    if nll_record.get("method") != checkpoint_context.method:
        raise ValueError(
            f"NLL evaluation stage method does not match the {endpoint_label} request"
        )
    if nll_record.get("checkpoint_context") != expected_context:
        raise ValueError(
            f"NLL evaluation checkpoint context does not match the {endpoint_label} request"
        )
    expected_execution = normalize_operator_identity(
        _json_object(execution, "execution")
    )
    recorded_execution = nll_record.get("execution")
    if checkpoint_context.method in GROUPED_BASE_METHODS:
        if not isinstance(recorded_execution, Mapping):
            raise ValueError("Grouped evaluation requires recorded execution evidence")
        expected_execution = dict(expected_execution)
        recorded_execution = dict(recorded_execution)
        quality_key = "grouped_piqa_quality_execution"
        reference_quality = expected_execution.pop(quality_key, None)
        recorded_reference_quality = recorded_execution.pop(quality_key, None)
        if recorded_reference_quality is not None:
            raise ValueError(
                "NLL evaluation cannot carry grouped PIQA reference execution"
            )
        if reference_quality is not None:
            quality_valid = _is_grouped_piqa_reference_quality_execution(
                reference_quality
            )
            if not quality_valid:
                raise ValueError(
                    "Grouped PIQA reference execution evidence is malformed"
                )
    if recorded_execution != expected_execution:
        raise ValueError(
            f"NLL evaluation execution does not match the {endpoint_label} request"
        )
    for label, expected_value in (
        ("data_identity", data_identity),
        ("model_identity", model_identity),
    ):
        if nll_record.get(label) != _json_object(expected_value, label):
            raise ValueError(
                f"NLL evaluation {label} does not match the {endpoint_label} request"
            )
    _json_object(source_identity, "source_identity")
    if nll_record.get("experiment_identity") != normalize_operator_identity(
        _json_object(experiment_identity, "experiment_identity")
    ):
        raise ValueError(
            f"NLL experiment identity does not match the {endpoint_label} request"
        )


def _is_grouped_piqa_reference_quality_execution(value: object) -> bool:
    """Recognize the sole explicit non-Triton endpoint allowed for grouped PIQA."""
    if isinstance(value, Mapping):
        value = normalize_operator_identity(value)
    return value == {
        "schema": "grouped_piqa_reference_quality_v1",
        "execution": "reference",
        "public_callable": "oal_attention.oal_attention",
        "scope": "quality_only_no_triton_capability_or_performance_claim",
        "max_sequence_length": 256,
    }


def _grouped_piqa_reference_alignment_seed(
    checkpoint_context: CheckpointContext,
) -> int:
    """Reuse the frozen diagnostic-input derivation rather than inventing a PIQA seed."""
    derivations = checkpoint_context.config_identity.get("seed_derivations")
    if not isinstance(derivations, Mapping):
        raise ValueError(
            "Grouped PIQA reference alignment requires frozen seed derivations"
        )
    seed = derivations.get("diagnostic_inputs_seed")
    if type(seed) is not int:
        raise ValueError(
            "Grouped PIQA reference alignment requires diagnostic_inputs_seed"
        )
    return seed


def _require_pilot_nll_prerequisite(nll_record: Mapping[str, object]) -> None:
    """Require the same-run NLL record without inventing a cohort-level claim."""
    if nll_record.get("evaluation_scope") != PILOT_EVALUATION_SCOPE:
        raise ValueError(
            "pilot PIQA evaluation requires a pilot_single_seed NLL record"
        )
    if "formal_evidence" in nll_record:
        raise ValueError(
            "pilot evaluation requires a same-run NLL record without cohort evidence"
        )


def _reject_production_evaluation_test_seams(
    checkpoint_context: EvaluationContext,
    *,
    best_loader: _BestLoader | None,
    model_forward: ModelForward | None,
) -> None:
    """Keep production checkpoint evidence on the native loader/forward path."""
    if checkpoint_context.config_identity.get("checkpoint_context_kind") != "formal":
        return
    if best_loader is not load_best_adapter:
        raise ValueError("production evaluation rejects injected best_loader")
    if model_forward is not None:
        raise ValueError("production evaluation rejects injected model_forward")
