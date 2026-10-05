"""Runtime identity construction and immutable evidence persistence."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from . import statistics_identity, data
from .protocol import STATISTICS_PROTOCOL_VERSION
from .config import GROUPED_BASE_METHODS, PilotConfig, TRAINABLE_KERNEL_METHODS
from .experiment_contract import NUM_LAYERS
from .grouped_provenance import _grouped_base_asset_runtime_identity
from .paths import _is_sha256, canonical_json
from .operator_imports import normalize_operator_identity
from .run_artifacts import (
    RUNTIME_EVIDENCE_FILENAME,
    _create_or_validate_immutable_json,
    _read_json_mapping,
)
from .telemetry import require_cohort_member_mutable, cohort_member_lock
from .workflows.errors import OrchestrationError

EXPERIMENT_PROTOCOL_VERSION = STATISTICS_PROTOCOL_VERSION
FLASH_KERNEL_PROTOCOL_VERSION = "flash-taylor-attn-public-grouped-quadratic-v1"
_FORMAL_GROUPED_EVIDENCE_HANDOFF = object()


def _write_or_validate_runtime_evidence(
    run_dir: Path, evidence: Mapping[str, object]
) -> None:
    """Persist fresh-run runtime identity and require exact semantic resume parity."""
    destination = run_dir / RUNTIME_EVIDENCE_FILENAME
    if not isinstance(evidence, Mapping):
        raise TypeError("runtime evidence must be a mapping")
    try:
        normalized = json.loads(canonical_json(normalize_operator_identity(evidence)))
    except ValueError as exc:
        raise OrchestrationError(
            "runtime evidence must be canonical JSON-safe"
        ) from exc
    if not isinstance(normalized, dict):
        raise OrchestrationError("runtime evidence must serialize to an object")
    try:
        with cohort_member_lock(run_dir):
            require_cohort_member_mutable(run_dir)
            if destination.is_file():
                stored = normalize_operator_identity(
                    _read_json_mapping(destination, "runtime evidence")
                )
                if canonical_json(stored) != canonical_json(normalized):
                    raise ValueError(
                        "runtime evidence does not match immutable run semantics"
                    )
                return
            _create_or_validate_immutable_json(
                destination, normalized, "runtime evidence"
            )
    except ValueError as exc:
        raise OrchestrationError(
            "runtime evidence does not match immutable run semantics"
        ) from exc


def _require_runtime_evidence(run_dir: Path, evidence: Mapping[str, object]) -> None:
    """Require final evaluation to reproduce the frozen training runtime identity."""
    destination = run_dir / RUNTIME_EVIDENCE_FILENAME
    if not destination.is_file():
        raise OrchestrationError(
            "final evaluation is missing immutable runtime evidence"
        )
    stored = normalize_operator_identity(
        _read_json_mapping(destination, "runtime evidence")
    )
    try:
        expected = json.loads(canonical_json(normalize_operator_identity(evidence)))
    except ValueError as exc:
        raise OrchestrationError(
            "runtime evidence must be canonical JSON-safe"
        ) from exc
    if canonical_json(stored) != canonical_json(expected):
        raise OrchestrationError(
            "final evaluation runtime evidence does not match frozen training semantics"
        )


def _frozen_protocol_identity(run_dir: Path) -> dict[str, object] | None:
    """Return the run's frozen statistics/methodology generation, if any.

    Post-hoc evaluation reproduces the generation under which the run's training
    evidence (``runtime_evidence.json``) was frozen, not the currently
    registered generation: ``STATISTICS_PROTOCOL_VERSION`` is a cohort marker
    that advances independently of one run's own semantics.  Returns exactly the
    protocol fields the frozen evidence records (kernel tuning evidence records only the
    version, never a LoRA statistics identity), or ``None`` when no frozen
    generation is recorded.
    """
    destination = run_dir / RUNTIME_EVIDENCE_FILENAME
    if not destination.is_file():
        return None
    stored = _read_json_mapping(destination, "runtime evidence")
    version = stored.get("experiment_protocol_version")
    if not isinstance(version, str) or not version:
        return None
    reconciled: dict[str, object] = {"experiment_protocol_version": version}
    statistics_identity = stored.get("statistics_identity")
    if isinstance(statistics_identity, Mapping):
        reconciled["statistics_identity"] = dict(statistics_identity)
    return reconciled


def _adopt_frozen_protocol_identity(
    run_dir: Path, identity: Mapping[str, object]
) -> dict[str, object]:
    """Adopt a pre-bump run's frozen statistics/methodology protocol identity.

    ``STATISTICS_PROTOCOL_VERSION`` is a cohort-generation marker advanced when
    the comparable method cohort changes; the comparison pipeline enforces it
    independently (``comparison._formal_statistics_identity``).  Advancing it
    does not change one run's own semantics, so it must not invalidate the
    frozen runtime evidence of an unrelated existing run.  Evaluation therefore
    adopts the run's frozen protocol fields so the evidence gate passes and any
    post-hoc output keeps the run's own training-generation label.
    """
    frozen = _frozen_protocol_identity(run_dir)
    if frozen is None:
        return dict(identity)
    reconciled = dict(identity)
    for field in ("experiment_protocol_version", "statistics_identity"):
        value = frozen.get(field)
        if isinstance(value, (str, Mapping)):
            reconciled[field] = value
    return reconciled


def _experiment_identity(
    config: PilotConfig,
    *,
    runtime: object,
    attention_execution: Mapping[str, object],
    effective_config_sha256: str,
    data_manifest_sha256: object,
    model_identity: Mapping[str, object],
    data_manifest: Mapping[str, object] | None = None,
    formal_grouped_admission: object | None = None,
    _formal_grouped_evidence_handoff: object | None = None,
) -> dict[str, object]:
    """Collect semantic runtime facts used by NLL/PIQA and cross-run gates."""
    expected_replacements = list(config.replacement_layer_ids)
    actual_replacements = attention_execution.get("replaced_layer_ids_zero_based")
    if actual_replacements != expected_replacements:
        raise OrchestrationError(
            "runtime attention replacement layers do not match the effective configuration"
        )
    kernel_bank = getattr(runtime, "kernel_bank", None)
    raw_kernel_layers = getattr(kernel_bank, "trainable_kernel_layer_ids", ())
    if callable(raw_kernel_layers):
        raw_kernel_layers = raw_kernel_layers()
    if not isinstance(raw_kernel_layers, Sequence) or isinstance(
        raw_kernel_layers, (str, bytes)
    ):
        raise OrchestrationError(
            "kernel bank trainable_kernel_layer_ids must be a sequence"
        )
    kernel_layers = list(raw_kernel_layers)
    if any((type(layer_id) is not int for layer_id in kernel_layers)):
        raise OrchestrationError(
            "kernel bank trainable_kernel_layer_ids must contain integers"
        )
    expected_kernel_layers = (
        expected_replacements if config.method in TRAINABLE_KERNEL_METHODS else []
    )
    if kernel_layers != expected_kernel_layers:
        raise OrchestrationError(
            "runtime trainable kernel layers do not match the method replacement contract"
        )
    if not isinstance(data_manifest_sha256, str) or not _is_sha256(
        data_manifest_sha256
    ):
        raise OrchestrationError("data manifest identity must be a lowercase SHA-256")
    identity: dict[str, object] = {
        "experiment_protocol_version": EXPERIMENT_PROTOCOL_VERSION,
        "statistics_identity": statistics_identity.statistics_identity(),
        "flash_kernel_protocol_version": FLASH_KERNEL_PROTOCOL_VERSION,
        "method": config.method,
        "method_identity": getattr(config, "method_identity", config.method),
        "tuning_mode": config.tuning_mode,
        "lora_layer_ids_zero_based": list(range(NUM_LAYERS)),
        "replaced_layer_ids_zero_based": expected_replacements,
        "trainable_kernel_layer_ids_zero_based": kernel_layers,
        "effective_config_sha256": effective_config_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "model_identity": dict(model_identity),
        "attention_execution": dict(attention_execution),
    }
    transformers_version = (
        getattr(runtime, "transformers_version", None)
        if config.model_profile is not None
        else None
    )
    if transformers_version is not None:
        if not isinstance(transformers_version, str) or not transformers_version:
            raise OrchestrationError(
                "runtime Transformers version must be a non-empty string"
            )
        identity["transformers_version"] = transformers_version
    if config.method not in GROUPED_BASE_METHODS:
        identity["formal_admission"] = {
            "status": "ready",
            "asset_status": "not_applicable",
        }
        return identity
    asset, grouped_asset_identity = _grouped_base_asset_runtime_identity(
        config, model_identity=model_identity, data_manifest=data_manifest
    )
    identity.update(grouped_asset_identity)
    identity["formal_admission"] = {
        "status": "precomputed_asset_loaded",
        "asset_status": "precomputed",
    }
    return identity


def _stable_model_identity(value: object) -> dict[str, object]:
    """Keep only path-free semantic model identity for lifecycle comparison."""
    if not isinstance(value, Mapping):
        raise OrchestrationError("model identity collector must return a mapping")
    identity = dict(value)
    identity.pop("collected_at", None)
    identity.pop("model_path", None)
    identity.pop("identity_sha256", None)
    if not identity:
        raise OrchestrationError(
            "model identity collector returned an empty stable identity"
        )
    return identity


def _assert_runtime_tokenizer_matches_manifest(
    config: PilotConfig, runtime: object, manifest: Mapping[str, object]
) -> None:
    """Reject a model tokenizer that cannot reproduce the prepared token blocks."""
    expected = manifest.get("tokenizer")
    if expected is None:
        return
    if not isinstance(expected, Mapping):
        raise OrchestrationError("data manifest tokenizer identity must be a mapping")
    tokenizer = getattr(runtime, "tokenizer", None)
    if tokenizer is None:
        raise OrchestrationError(
            "model runtime must provide tokenizer identity for prepared assets"
        )
    from .data import tokenizer_identity

    actual = tokenizer_identity(
        tokenizer,
        vocab_file=data._local_tokenizer_file(config.model_path, "tokenizer.json"),
        config_file=data._local_tokenizer_file(
            config.model_path, "tokenizer_config.json"
        ),
    )
    if canonical_json(actual) != canonical_json(dict(expected)):
        raise OrchestrationError(
            "runtime tokenizer identity does not match prepared data manifest"
        )
