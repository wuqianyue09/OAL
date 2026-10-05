"""Checkpoint schema versions, run identity contexts, and model-scope validation."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Literal, TypeAlias
from .config import METHOD_NAMES, PilotConfig, TRAINABLE_KERNEL_METHODS
from .paths import canonical_json
from .protocol import (
    BOOTSTRAP_RESAMPLES,
    REGISTERED_STATISTICS_PROTOCOL_VERSIONS,
    SEED_PROTOCOL_VERSION,
    derive_endpoint_bootstrap_seed,
    validate_persisted_seed_derivations,
)

CHECKPOINT_SCHEMA_VERSION = 2
LATEST_RESUME_FILENAME = "latest_resume.pt"
CheckpointKind = Literal["best_adapter", "latest_resume"]
JsonScalar: TypeAlias = None | bool | int | float | str
_MODEL_SCOPE_FIELDS = frozenset(
    (
        "model_profile",
        "model_geometry",
        "requested_replacement_layer_ids_zero_based",
        "replaced_layer_ids_zero_based",
        "lora_layer_ids_zero_based",
        "trainable_kernel_layer_ids_zero_based",
    )
)
_HISTORICAL_FORMAL_SCOPE_FIELDS = frozenset(
    ("replaced_layer_ids_zero_based", "lora_layer_ids_zero_based")
)


def add_model_scope_identity(
    config: PilotConfig,
    *,
    config_identity: Mapping[str, object],
    model_identity: Mapping[str, object],
    attention_execution: Mapping[str, object],
    kernel_bank: object,
    allow_missing_kernel_scope: bool = False,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Bind one checkpoint identity to the assembled model and its exact scopes."""
    if not isinstance(config, PilotConfig):
        raise TypeError("config must be a PilotConfig")
    config.validate()
    normalized_config = _normalise_metadata_mapping(config_identity, "config_identity")
    normalized_model = _normalise_metadata_mapping(model_identity, "model_identity")
    normalized_execution = _normalise_metadata_mapping(
        attention_execution, "attention_execution"
    )
    actual_replacements = normalized_execution.get("replaced_layer_ids_zero_based")
    if actual_replacements != list(config.replacement_layer_ids):
        raise ValueError(
            "attention execution replacement layers do not match the effective configuration"
        )
    raw_kernel_layers = getattr(kernel_bank, "trainable_kernel_layer_ids", None)
    if callable(raw_kernel_layers):
        raw_kernel_layers = raw_kernel_layers()
    if raw_kernel_layers is None and allow_missing_kernel_scope:
        raw_kernel_layers = config.trainable_kernel_layer_ids
    if not isinstance(raw_kernel_layers, (list, tuple)) or any(
        (type(layer_id) is not int for layer_id in raw_kernel_layers)
    ):
        raise ValueError("kernel bank trainable layers must be an integer sequence")
    kernel_layers = list(raw_kernel_layers)
    if kernel_layers != list(config.trainable_kernel_layer_ids):
        raise ValueError(
            "kernel bank trainable layers do not match the effective configuration"
        )
    geometry = asdict(config.geometry)
    topology = {
        "model_profile": config.resolved_model_profile,
        "model_geometry": geometry,
        "requested_replacement_layer_ids_zero_based": list(config.replacement_layers),
        "replaced_layer_ids_zero_based": list(actual_replacements),
        "lora_layer_ids_zero_based": list(config.scope.lora_layers),
        "trainable_kernel_layer_ids_zero_based": kernel_layers,
    }
    normalized_config.update(topology)
    normalized_model.update(
        {"model_profile": config.resolved_model_profile, "model_geometry": geometry}
    )
    normalized_execution["model_profile"] = config.resolved_model_profile
    normalized_execution["model_geometry"] = geometry
    return (normalized_config, normalized_model, normalized_execution)


def normalize_legacy_qwen_identity(
    identity: Mapping[str, object],
    *,
    model: object,
    expected_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Interpret one old profile-less identity only for an actual Qwen runtime."""
    normalized = _normalise_metadata_mapping(identity, "identity")
    if "model_profile" in normalized:
        return normalized
    model_config = getattr(model, "config", None)
    if model_config is None:
        model_config = getattr(getattr(model, "model", None), "config", None)
    expected_geometry = {
        "num_layers": 24,
        "hidden_size": 896,
        "num_query_heads": 14,
        "num_kv_heads": 2,
        "head_dim": 64,
    }
    actual_geometry = {
        "num_layers": getattr(model_config, "num_hidden_layers", None),
        "hidden_size": getattr(model_config, "hidden_size", None),
        "num_query_heads": getattr(model_config, "num_attention_heads", None),
        "num_kv_heads": getattr(model_config, "num_key_value_heads", None),
        "head_dim": getattr(model_config, "head_dim", None),
    }
    if actual_geometry["head_dim"] is None:
        hidden_size = actual_geometry["hidden_size"]
        query_heads = actual_geometry["num_query_heads"]
        if type(hidden_size) is int and type(query_heads) is int and query_heads:
            actual_geometry["head_dim"] = hidden_size // query_heads
    if (
        getattr(model_config, "model_type", None) != "qwen2"
        or actual_geometry != expected_geometry
    ):
        raise ValueError(
            "profile-less checkpoint identity is legacy Qwen only, but the live model is not the Qwen2.5-0.5B geometry"
        )
    normalized["model_profile"] = "qwen2_5_0_5b"
    normalized["model_geometry"] = expected_geometry
    if "method" in normalized:
        method = normalized.get("method")
        expected_requested = (
            expected_identity.get("requested_replacement_layer_ids_zero_based")
            if isinstance(expected_identity, Mapping)
            else None
        )
        replaced = normalized.get("replaced_layer_ids_zero_based")
        if not (
            isinstance(expected_requested, list)
            and all((type(layer_id) is int for layer_id in expected_requested))
        ):
            expected_requested = None
        requested = (
            list(replaced)
            if isinstance(replaced, list)
            else (
                expected_requested
                if expected_requested is not None
                else list(range(3, 21))
            )
        )
        normalized.setdefault("replaced_layer_ids_zero_based", list(requested))
        normalized.setdefault(
            "requested_replacement_layer_ids_zero_based", list(requested)
        )
        normalized.setdefault("lora_layer_ids_zero_based", list(range(24)))
        normalized.setdefault(
            "trainable_kernel_layer_ids_zero_based",
            (
                list(normalized["replaced_layer_ids_zero_based"])
                if method in TRAINABLE_KERNEL_METHODS
                else []
            ),
        )
    return normalized


@dataclass(frozen=True)
class CheckpointContext:
    """Stable run facts that every save and load must match exactly.

    Callers must pass the *stable* effective configuration and data/model
    identities.  Timestamps and other process-specific observations belong in
    telemetry, not here: this class intentionally compares values exactly and
    never tries to guess which metadata fields are safe to ignore.
    """

    method: str
    config_identity: Mapping[str, object]
    data_identity: Mapping[str, object]
    model_identity: Mapping[str, object]

    def __post_init__(self) -> None:
        _require_method(self.method)
        object.__setattr__(
            self,
            "config_identity",
            _normalise_metadata_mapping(self.config_identity, "config_identity"),
        )
        object.__setattr__(
            self,
            "data_identity",
            _normalise_metadata_mapping(self.data_identity, "data_identity"),
        )
        object.__setattr__(
            self,
            "model_identity",
            _normalise_metadata_mapping(self.model_identity, "model_identity"),
        )
        context_kind = self.config_identity.get("checkpoint_context_kind")
        if context_kind not in {"formal", "nonformal_test"}:
            raise ValueError(
                "config_identity.checkpoint_context_kind must be 'formal' or 'nonformal_test'"
            )
        if context_kind == "formal":
            _validate_formal_checkpoint_context(
                method=self.method,
                config_identity=self.config_identity,
                data_identity=self.data_identity,
                model_identity=self.model_identity,
            )


def _require_method(value: object) -> str:
    if not isinstance(value, str) or value not in METHOD_NAMES:
        raise ValueError(f"method must be one of {METHOD_NAMES}")
    return value


_GROUPED_PUBLIC_CALLABLE = "oal_attention.oal_attention"


def _validate_model_scope_identity(
    *,
    method: str,
    config_identity: Mapping[str, object],
    model_identity: Mapping[str, object],
) -> None:
    """Validate the profile/topology block when reading a new-format identity."""
    present = _MODEL_SCOPE_FIELDS & set(config_identity)
    if not present:
        return
    if (
        config_identity.get("checkpoint_context_kind") == "formal"
        and present <= _HISTORICAL_FORMAL_SCOPE_FIELDS
        and (not {"model_profile", "model_geometry"} & set(model_identity))
    ):
        return
    missing = sorted(_MODEL_SCOPE_FIELDS - set(config_identity))
    if missing:
        raise ValueError(
            "config_identity model/scope identity is incomplete: " + ", ".join(missing)
        )
    profile = config_identity["model_profile"]
    if profile not in {"qwen2_5_0_5b", "llama3_2_1b_base"}:
        raise ValueError("config_identity.model_profile is unsupported")
    expected_geometry = {
        "qwen2_5_0_5b": {
            "num_layers": 24,
            "hidden_size": 896,
            "num_query_heads": 14,
            "num_kv_heads": 2,
            "head_dim": 64,
        },
        "llama3_2_1b_base": {
            "num_layers": 16,
            "hidden_size": 2048,
            "num_query_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 64,
        },
    }[profile]
    if config_identity["model_geometry"] != expected_geometry:
        raise ValueError("config_identity.model_geometry does not match model_profile")
    if model_identity.get("model_profile") != profile:
        raise ValueError("model_identity.model_profile does not match config_identity")
    if model_identity.get("model_geometry") != expected_geometry:
        raise ValueError("model_identity.model_geometry does not match config_identity")

    def layers(field: str) -> list[int]:
        value = config_identity[field]
        if (
            not isinstance(value, list)
            or any((type(layer_id) is not int for layer_id in value))
            or value != sorted(set(value))
            or any(
                (
                    layer_id < 0 or layer_id >= expected_geometry["num_layers"]
                    for layer_id in value
                )
            )
        ):
            raise ValueError(f"config_identity.{field} is not a valid layer scope")
        return value

    requested = layers("requested_replacement_layer_ids_zero_based")
    replaced = layers("replaced_layer_ids_zero_based")
    lora_layers = layers("lora_layer_ids_zero_based")
    kernel_layers = layers("trainable_kernel_layer_ids_zero_based")
    if replaced != requested:
        raise ValueError("config_identity actual replacement scope is inconsistent")
    expected_lora_layers = list(range(expected_geometry["num_layers"]))
    if lora_layers != expected_lora_layers:
        raise ValueError("config_identity LoRA scope is inconsistent with tuning mode")
    expected_kernel = replaced if method in TRAINABLE_KERNEL_METHODS else []
    if kernel_layers != expected_kernel:
        raise ValueError("config_identity trainable kernel scope is inconsistent")


def _validate_formal_checkpoint_context(
    *,
    method: str,
    config_identity: Mapping[str, object],
    data_identity: Mapping[str, object],
    model_identity: Mapping[str, object],
) -> None:
    """Fail closed at the checkpoint boundary for formal run evidence."""
    if config_identity.get("method") != method:
        raise ValueError(
            "formal config_identity.method does not match checkpoint method"
        )
    _validate_model_scope_identity(
        method=method, config_identity=config_identity, model_identity=model_identity
    )
    profile = config_identity.get("model_profile", "qwen2_5_0_5b")
    master_seed = config_identity.get("master_seed")
    if type(master_seed) is not int:
        raise ValueError("formal config_identity.master_seed must be an integer")
    if config_identity.get("seed_protocol_version") != SEED_PROTOCOL_VERSION:
        raise ValueError(
            "formal config_identity seed protocol version is missing or unsupported"
        )
    try:
        derivations = validate_persisted_seed_derivations(
            config_identity.get("seed_derivations"), master_seed=master_seed
        )
    except ValueError as exc:
        raise ValueError(
            "formal config_identity seed protocol derivations are invalid"
        ) from exc
    seed_map_sha256 = config_identity.get("seed_map_sha256")
    expected_seed_map_sha256 = hashlib.sha256(
        canonical_json(derivations.to_dict()).encode("utf-8")
    ).hexdigest()
    if seed_map_sha256 != expected_seed_map_sha256:
        raise ValueError(
            "formal config_identity seed_map_sha256 does not match seed derivations"
        )
    lora_hashes = config_identity.get("lora_initial_parameter_sha256")
    lora_hashes_sha256 = config_identity.get("lora_initial_hashes_sha256")
    if not isinstance(lora_hashes, Mapping) or not lora_hashes:
        raise ValueError(
            "formal config_identity must contain LoRA initial parameter hashes"
        )
    normalized_lora_hashes = _normalise_metadata_mapping(
        lora_hashes, "lora_initial_parameter_sha256"
    )
    if not all(
        (
            isinstance(name, str) and _is_sha256(digest)
            for (name, digest) in normalized_lora_hashes.items()
        )
    ):
        raise ValueError("formal LoRA initial parameter hashes must be SHA-256 digests")
    expected_lora_hashes_sha256 = hashlib.sha256(
        canonical_json(normalized_lora_hashes).encode("utf-8")
    ).hexdigest()
    if lora_hashes_sha256 != expected_lora_hashes_sha256:
        raise ValueError(
            "formal LoRA initial hash aggregate does not match parameter hashes"
        )
    statistics_identity = config_identity.get("statistics_identity")
    statistics_version = config_identity.get("experiment_protocol_version")
    if statistics_version not in REGISTERED_STATISTICS_PROTOCOL_VERSIONS:
        raise ValueError(
            "formal config_identity statistics protocol version is not registered"
        )
    expected_statistics_identity = {
        "statistics_protocol_version": statistics_version,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "endpoint_bootstrap_seeds": {
            "test_nll": derive_endpoint_bootstrap_seed(
                statistics_protocol_version=statistics_version,
                endpoint_namespace="test_nll",
            ),
            "piqa_acc_norm": derive_endpoint_bootstrap_seed(
                statistics_protocol_version=statistics_version,
                endpoint_namespace="piqa_acc_norm",
            ),
        },
    }
    if statistics_identity != expected_statistics_identity:
        raise ValueError(
            "formal config_identity statistics identity is not fixed by the protocol"
        )
    if not _is_sha256(config_identity.get("runtime_evidence_sha256")):
        raise ValueError(
            "formal config_identity runtime evidence SHA-256 is missing or invalid"
        )
    if not model_identity:
        raise ValueError("formal model_identity must not be empty")
    required_data_fields = {
        "manifest_sha256",
        "sequence_length",
        "train_permutation_master_seed",
        "train_permutation_seed",
        "train_permutation_sha256",
        "train_selected_blocks_sha256",
        "train_permutation_identity",
        "validation_block_indices",
        "validation_block_indices_sha256",
        "validation_selected_blocks_sha256",
    }
    missing = sorted(required_data_fields - set(data_identity))
    if missing:
        raise ValueError(
            "formal data_identity is missing required seed/data-selection evidence: "
            + ", ".join(missing)
        )
    for field in (
        "manifest_sha256",
        "train_permutation_sha256",
        "train_selected_blocks_sha256",
        "train_permutation_identity",
        "validation_block_indices_sha256",
        "validation_selected_blocks_sha256",
    ):
        if not _is_sha256(data_identity[field]):
            raise ValueError(
                f"formal data_identity.{field} must be a SHA-256 hex string"
            )
    if data_identity["train_permutation_master_seed"] != master_seed:
        raise ValueError(
            "formal data permutation master seed does not match config master seed"
        )
    if data_identity["train_permutation_seed"] != derivations.data_permutation_seed:
        raise ValueError("formal data permutation seed does not match seed derivations")
    validation_indices = data_identity["validation_block_indices"]
    if (
        not isinstance(validation_indices, list)
        or any((type(index) is not int for index in validation_indices))
        or validation_indices != list(range(len(validation_indices)))
    ):
        raise ValueError(
            "formal validation_block_indices must be canonical zero-based indices"
        )
    validation_blocks = config_identity.get("validation_blocks")
    if type(validation_blocks) is not int or validation_blocks < 0:
        raise ValueError(
            "formal config_identity.validation_blocks must be a non-negative integer"
        )
    if len(validation_indices) != validation_blocks:
        raise ValueError(
            "formal validation selection length does not match config identity"
        )
    if data_identity["sequence_length"] != config_identity.get("sequence_length"):
        raise ValueError("formal data sequence length does not match config identity")
    validation_indices_sha = hashlib.sha256(
        canonical_json(validation_indices).encode("utf-8")
    ).hexdigest()
    if data_identity["validation_block_indices_sha256"] != validation_indices_sha:
        raise ValueError(
            "formal validation_block_indices SHA-256 does not match indices"
        )
    expected_train_identity = hashlib.sha256(
        canonical_json(
            {
                "master_seed": master_seed,
                "permutation_seed": data_identity["train_permutation_seed"],
                "permutation_sha256": data_identity["train_permutation_sha256"],
                "selected_train_blocks_sha256": data_identity[
                    "train_selected_blocks_sha256"
                ],
            }
        ).encode("utf-8")
    ).hexdigest()
    if data_identity["train_permutation_identity"] != expected_train_identity:
        raise ValueError(
            "formal train permutation identity does not match selection evidence"
        )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all((character in "0123456789abcdef" for character in value))
    )


def _normalise_metadata_mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping with string keys")
    keys = list(value)
    for key in keys:
        _require_metadata_key(key, field)
    from .operator_imports import normalize_operator_identity

    return normalize_operator_identity(
        {
            key: _normalise_metadata(item, f"{field}.{key}")
            for (key, item) in sorted(value.items(), key=lambda pair: pair[0])
        }
    )


def _require_metadata_key(key: object, field: str) -> bool:
    if not isinstance(key, str) or not key:
        raise ValueError(f"{field} metadata keys must be non-empty strings")
    return True


def _normalise_metadata(value: object, field: str) -> object:
    if value is None or isinstance(value, (bool, str)):
        return value
    if type(value) is int:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} must not contain non-finite metadata")
        return value
    if isinstance(value, Mapping):
        return _normalise_metadata_mapping(value, field)
    if isinstance(value, (list, tuple)):
        return [
            _normalise_metadata(item, f"{field}[{index}]")
            for (index, item) in enumerate(value)
        ]
    raise ValueError(
        f"{field} metadata must contain only JSON-compatible scalar, list, or mapping values"
    )
