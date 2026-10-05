"""Checkpoint context and immutable initial LoRA evidence."""

from __future__ import annotations
from collections.abc import Mapping
import json
from pathlib import Path
from . import statistics_identity as stats_identity, run_artifacts
from .config import GROUPED_BASE_METHODS, PilotConfig
from .experiment_contract import FORMAL_MASTER_SEEDS, NUM_LAYERS
from .paths import _is_sha256, SCHEMA_VERSION, canonical_json, sha256_bytes, sha256_file
from .protocol import SEED_PROTOCOL_VERSION
from .run_artifacts import (
    DATA_MANIFEST_COPY_FILENAME,
    EFFECTIVE_CONFIG_FILENAME,
    LORA_INITIAL_HASHES_FILENAME,
    _create_or_validate_immutable_json,
)
from .runtime_identity import EXPERIMENT_PROTOCOL_VERSION, FLASH_KERNEL_PROTOCOL_VERSION
from .telemetry import require_cohort_member_mutable, cohort_member_lock
from .workflows.errors import OrchestrationError


def _load_or_create_lora_initial_hashes_identity(
    run_dir: Path, *, model: object | None, formal: bool, create: bool
) -> dict[str, object] | None:
    """Load the immutable pre-training LoRA tensor hashes for formal evidence."""
    if not formal:
        return None
    destination = run_dir / LORA_INITIAL_HASHES_FILENAME
    if not create:
        if not destination.exists():
            raise OrchestrationError(
                "formal run is missing immutable LoRA initial hash evidence"
            )
        record = run_artifacts._safe_run_artifact_json(
            run_dir, LORA_INITIAL_HASHES_FILENAME, "LoRA initial hash evidence"
        )
        return _lora_initial_hashes_identity_from_record(record)
    with cohort_member_lock(run_dir):
        require_cohort_member_mutable(run_dir)
        if destination.exists():
            record = run_artifacts._safe_run_artifact_json(
                run_dir, LORA_INITIAL_HASHES_FILENAME, "LoRA initial hash evidence"
            )
            return _lora_initial_hashes_identity_from_record(record)
        if model is None:
            raise OrchestrationError(
                "formal LoRA initial hash creation requires the fresh model"
            )
        record = _build_lora_initial_hashes_record(model)
        _create_or_validate_immutable_json(
            destination, record, "LoRA initial hash evidence"
        )
        return _lora_initial_hashes_identity_from_record(
            run_artifacts._safe_run_artifact_json(
                run_dir, LORA_INITIAL_HASHES_FILENAME, "LoRA initial hash evidence"
            )
        )


def _build_lora_initial_hashes_record(model: object) -> dict[str, object]:
    """Hash every injected LoRA A/B tensor before the first training step."""
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise OrchestrationError(
            "formal model does not expose named_parameters for LoRA hash evidence"
        )
    try:
        import torch
    except ImportError as exc:
        raise OrchestrationError("formal LoRA initial hashes require torch") from exc
    parameter_hashes: dict[str, str] = {}
    for name, parameter in named_parameters():
        if not isinstance(name, str) or not (
            name.endswith(".A") or name.endswith(".B")
        ):
            continue
        if not isinstance(parameter, torch.Tensor):
            raise OrchestrationError("LoRA initial parameter is not a tensor")
        raw = (
            parameter.detach()
            .to(device="cpu")
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
        parameter_hashes[name] = sha256_bytes(raw)
    if not parameter_hashes:
        raise OrchestrationError(
            "formal model has no LoRA A/B parameters for initial hash evidence"
        )
    parameter_hashes = dict(sorted(parameter_hashes.items()))
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "qwen_lora_initial_hashes",
        "parameter_sha256": parameter_hashes,
        "parameter_hashes_sha256": sha256_bytes(
            canonical_json(parameter_hashes).encode("utf-8")
        ),
    }


def _lora_initial_hashes_identity_from_record(
    record: Mapping[str, object],
) -> dict[str, object]:
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("kind") != "qwen_lora_initial_hashes"
    ):
        raise OrchestrationError(
            "LoRA initial hash evidence has an incompatible schema or kind"
        )
    hashes = record.get("parameter_sha256")
    aggregate = record.get("parameter_hashes_sha256")
    if not isinstance(hashes, Mapping) or not hashes:
        raise OrchestrationError("LoRA initial hash evidence lacks parameter hashes")
    normalized = json.loads(canonical_json(dict(hashes)))
    if not isinstance(normalized, dict) or not all(
        (
            isinstance(name, str) and _is_sha256(digest)
            for (name, digest) in normalized.items()
        )
    ):
        raise OrchestrationError("LoRA initial parameter hashes are invalid")
    expected = sha256_bytes(canonical_json(normalized).encode("utf-8"))
    if aggregate != expected:
        raise OrchestrationError("LoRA initial hash evidence aggregate is inconsistent")
    return {
        "lora_initial_parameter_sha256": normalized,
        "lora_initial_hashes_sha256": aggregate,
    }


def _checkpoint_config_identity(
    config: PilotConfig,
    run_dir: Path,
    *,
    formal: bool,
    runtime_evidence_sha256: str | None = None,
    lora_initial_identity: Mapping[str, object] | None = None,
    protocol_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return the semantic checkpoint contract, never a whole-tree source hash.

    ``protocol_identity`` optionally supplies the run's frozen statistics
    generation (see :func:`_frozen_protocol_identity`).  Post-hoc evaluation of
    a pre-bump run must keep the run's own training-generation label, so the
    cohort marker is taken from the frozen evidence when present instead of the
    current global constant.
    """
    protocol_version = EXPERIMENT_PROTOCOL_VERSION
    statistics_identity: object = stats_identity.statistics_identity()
    if protocol_identity is not None:
        frozen_version = protocol_identity.get("experiment_protocol_version")
        frozen_statistics = protocol_identity.get("statistics_identity")
        if isinstance(frozen_version, str) and frozen_version:
            protocol_version = frozen_version
        if isinstance(frozen_statistics, Mapping) and frozen_statistics:
            statistics_identity = dict(frozen_statistics)
        elif protocol_version != EXPERIMENT_PROTOCOL_VERSION:
            statistics_identity = stats_identity.statistics_identity_for_version(
                protocol_version
            )
    effective_config_sha256 = (
        run_artifacts._safe_run_artifact_sha256(run_dir, EFFECTIVE_CONFIG_FILENAME)
        if formal
        else sha256_file(run_dir / EFFECTIVE_CONFIG_FILENAME)
    )
    identity = {
        "checkpoint_context_kind": "formal" if formal else "nonformal_test",
        "effective_config_sha256": effective_config_sha256,
        "method": config.method,
        "master_seed": config.seed,
        "tuning_mode": config.tuning_mode,
        "method_identity": getattr(config, "method_identity", config.method),
        "sequence_length": config.sequence_length,
        "validation_blocks": config.validation_blocks,
        "lora_layer_ids_zero_based": list(range(NUM_LAYERS)),
        "replaced_layer_ids_zero_based": list(config.replacement_layer_ids),
        "experiment_protocol_version": protocol_version,
        "statistics_identity": statistics_identity,
        "flash_kernel_protocol_version": FLASH_KERNEL_PROTOCOL_VERSION,
        "pre_registered_master_seeds": list(
            getattr(config, "pre_registered_master_seeds", FORMAL_MASTER_SEEDS)
        ),
        "seed_protocol_version": SEED_PROTOCOL_VERSION,
        "seed_derivations": config.seed_derivations.to_dict(),
    }
    if config.method in GROUPED_BASE_METHODS:
        identity["group_asset_sha256"] = config.group_asset_sha256 or sha256_file(
            config.group_asset_path
        )
    if formal:
        if not isinstance(lora_initial_identity, Mapping):
            raise OrchestrationError(
                "formal checkpoint identity requires immutable LoRA initial hashes"
            )
        identity.update(dict(lora_initial_identity))
        identity["seed_map_sha256"] = sha256_bytes(
            canonical_json(identity["seed_derivations"]).encode("utf-8")
        )
        identity["statistics_identity"] = statistics_identity
        if not _is_sha256(runtime_evidence_sha256):
            raise OrchestrationError(
                "formal checkpoint identity requires immutable runtime evidence SHA-256"
            )
        identity["runtime_evidence_sha256"] = runtime_evidence_sha256
    return identity


def _checkpoint_data_identity(
    config: PilotConfig, *, run_dir: Path, assets: object, formal: bool
) -> dict[str, object]:
    """Bind a formal checkpoint to its selected train/validation block order.

    ``TrainingAssets`` exposes these facts after schema-v2 validation.  The
    deliberately small injected-runtime seams used by CPU orchestration tests
    may omit all three fields; those seams remain non-production fixtures and
    retain the historical minimal identity.
    """
    identity: dict[str, object] = {
        "manifest_sha256": (
            run_artifacts._safe_run_artifact_sha256(
                run_dir, DATA_MANIFEST_COPY_FILENAME
            )
            if formal
            else sha256_file(run_dir / DATA_MANIFEST_COPY_FILENAME)
        ),
        "sequence_length": config.sequence_length,
    }
    if formal:
        from .assets import formal_training_selection_evidence

        try:
            identity.update(formal_training_selection_evidence(config, assets))
        except (TypeError, ValueError) as exc:
            raise OrchestrationError(
                "formal checkpoint data identity requires cryptographically verified native schema-v2 TrainingAssets selection"
            ) from exc
        return identity
    values = (
        getattr(assets, "train_permutation_seed", None),
        getattr(assets, "train_permutation_sha256", None),
        getattr(assets, "validation_block_indices", None),
    )
    if values == (None, None, None):
        return identity
    permutation_seed, permutation_sha256, validation_indices = values
    expected_seed = config.seed_derivations.data_permutation_seed
    if type(permutation_seed) is not int or permutation_seed != expected_seed:
        raise OrchestrationError(
            "training assets selected a permutation seed inconsistent with the formal config"
        )
    if not _is_sha256(permutation_sha256):
        raise OrchestrationError(
            "training assets selected permutation SHA-256 is invalid"
        )
    expected_validation_indices = tuple(range(config.validation_blocks))
    if (
        not isinstance(validation_indices, tuple)
        or validation_indices != expected_validation_indices
    ):
        raise OrchestrationError(
            "training assets validation block selection does not match the formal config"
        )
    validation_indices_list = list(validation_indices)
    identity.update(
        {
            "train_permutation_master_seed": config.seed,
            "train_permutation_seed": permutation_seed,
            "train_permutation_sha256": permutation_sha256,
            "validation_block_indices": validation_indices_list,
            "validation_block_indices_sha256": sha256_bytes(
                canonical_json(validation_indices_list).encode("utf-8")
            ),
        }
    )
    return identity
