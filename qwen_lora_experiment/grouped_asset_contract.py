"""Grouped asset representation, schema constants, and shape validation."""

from __future__ import annotations
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from .experiment_contract import (
    HEAD_DIM,
    NUM_LAYERS,
    NUM_QUERY_HEADS,
    REPLACEMENT_LAYER_IDS,
)
import json
import math
import numpy as np
from .asset_io import _validate_sha256

GROUP_ASSET_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
GROUP_ASSET_LAYERS = NUM_LAYERS
GROUP_ASSET_QUERY_HEADS = NUM_QUERY_HEADS
GROUP_ASSET_HEAD_DIM = HEAD_DIM
GROUP_ASSET_SEQUENCE_LENGTH = 4096
GROUP_ASSET_BINARY_DEFINITION = "group_0 versus union(group_1..group_K-1)"
STABLE_GAP_GROUPING_PROTOCOL = "qk_stable_gap_v1"
SIZE_MATCHED_RANDOM_GROUPING_PROTOCOL = "qk_size_matched_random_v1"
CANONICAL_ADAPTIVE_SCHEMA = "canonical_active_only_v2"
LEGACY_ADAPTIVE_SCHEMA = "legacy_v1_layer_padded"
ADAPTIVE_MULTIGROUP_METHOD = "adaptive_multigroup_full_quadratic"
ADAPTIVE_QUADRATIC_PARAMETERIZATION = "epsilon + ||[1,z_1,...,z_K] L||^2"
ADAPTIVE_QUADRATIC_OBJECTIVE = "direct real-V single-head causal attention output MSE"
LEGACY_ADAPTIVE_QUADRATIC_OBJECTIVE = "direct real-V single-head attention output MSE"
CANONICAL_ADAPTIVE_SEQUENCE_LENGTH = 2048
ADAPTIVE_GROUP_COUNT_RANGE = range(2, 6)
ADAPTIVE_MAX_STORED_GMAX = 8
STABLE_GAP_TOP_FRACTION = 0.25
STABLE_GAP_MINIMUM_ADJACENT_RATIO = 1.5
STABLE_GAP_MINIMUM_BOOTSTRAP_SUPPORT = 0.9
STABLE_GAP_TIE_BREAK = (
    "bootstrap_support_desc_then_adjacent_ratio_desc_then_cut_index_asc"
)
FACTOR_SELECTION_TIE_BREAK = "minimum_validation_mse_with_tolerance_then_initialization_then_earlier_checkpoint_then_factor_bytes"
_LEGACY_FACTOR_SELECTION_TIE_BREAK = "minimum_validation_mse_then_earlier_checkpoint_then_initialization_then_factor_bytes"
FORMAL_FACTOR_CALIBRATION_EVIDENCE_SCHEMA = "hd_mgq_factor_evidence_v1"
CANONICAL_ASSET_STATUS_FORMAL_PRIMARY = "formal_primary"
CANONICAL_ASSET_STATUS_LEGACY_TRANSFER = "legacy_transfer_initialization"
CANONICAL_ASSET_STATUS_NONFORMAL_CPU = "nonformal_cpu_fixture"
NONFORMAL_CPU_HARNESS_SEQUENCE_LENGTH = 128
CANONICAL_PRIMARY_LAYER_IDS = REPLACEMENT_LAYER_IDS
_CANONICAL_ADAPTIVE_ROOT_KEYS = {
    "schema",
    "asset_status",
    "model_name",
    "model_num_layers",
    "num_query_heads",
    "head_dim",
    "sequence_length",
    "method",
    "parameterization",
    "objective",
    "epsilon",
    "layer_ids_zero_based",
    "head_record_count",
    "heads",
    "provenance",
}
_CANONICAL_ADAPTIVE_HEAD_REQUIRED_KEYS = {
    "layer_id",
    "head_id",
    "group_count",
    "groups",
    "packed_lower_triangular",
    "cut_evidence",
    "factor_selection",
}
_CANONICAL_ADAPTIVE_HEAD_OPTIONAL_KEYS = {"validation_output_mse"}
_ADAPTIVE_PROVENANCE_REQUIRED_KEYS = {
    "model_identity",
    "tokenizer_identity",
    "dataset_identity",
    "grouping",
    "factor_calibration",
    "overlap_check",
    "semantic_summary",
}
_LEGACY_ADAPTIVE_ROOT_REQUIRED_KEYS = {
    "model_name",
    "layer_ids_zero_based",
    "head_dim",
    "epsilon",
    "seq_len",
    "method",
    "objective",
    "parameterization",
    "train_indices",
    "validation_indices",
    "test_indices",
    "layers",
}
_LEGACY_ADAPTIVE_ROOT_OPTIONAL_KEYS = {"schema", "minimum_group_count"}
_LEGACY_ADAPTIVE_HEAD_REQUIRED_KEYS = {
    "group_count",
    "groups",
    "packed_lower_triangular",
    "expanded_quadratic_matrix",
    "validation_output_mse",
}
_LEGACY_ADAPTIVE_HEAD_OPTIONAL_KEYS = {"special_dims"}
_GROUP_ASSET_ROOT_KEYS = {
    "binary_definition",
    "epsilon",
    "head_dim",
    "layer_ids_zero_based",
    "layers",
    "method",
    "model_name",
    "objective",
    "seq_len",
    "test_indices",
    "train_indices",
    "validation_indices",
}
_GROUP_ASSET_HEAD_KEYS = {"parameters", "special_dims", "validation_output_mse"}
_GROUP_PARAMETER_KEYS = {"u0", "u1", "u2", "v0", "v1", "v2"}


@dataclass(frozen=True)
class AdaptiveGroupHeadRecord:
    """One canonical active-only factor and its exact dimension partition."""

    layer_id: int
    head_id: int
    group_count: int
    groups: tuple[tuple[int, ...], ...]
    packed_lower_triangular: tuple[float, ...]
    cut_evidence: Mapping[str, object]
    factor_selection: Mapping[str, object]
    validation_output_mse: float | None = None


@dataclass(frozen=True)
class CanonicalAdaptiveGroupAsset:
    """Validated, active-only records for exactly the requested model layers."""

    schema: str
    asset_status: str
    model_name: str
    model_num_layers: int
    num_query_heads: int
    head_dim: int
    sequence_length: int
    method: str
    parameterization: str
    objective: str
    epsilon: float
    layer_ids_zero_based: tuple[int, ...]
    heads: Mapping[tuple[int, int], AdaptiveGroupHeadRecord]
    provenance: Mapping[str, object]
    source_sha256: str | None = None

    @property
    def head_record_count(self) -> int:
        return len(self.heads)

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-ready canonical representation in deterministic order."""
        records: list[dict[str, object]] = []
        for key in sorted(self.heads):
            record = self.heads[key]
            raw_record: dict[str, object] = {
                "layer_id": record.layer_id,
                "head_id": record.head_id,
                "group_count": record.group_count,
                "groups": [list(group) for group in record.groups],
                "packed_lower_triangular": list(record.packed_lower_triangular),
                "cut_evidence": _json_copy(dict(record.cut_evidence)),
                "factor_selection": _json_copy(dict(record.factor_selection)),
            }
            if record.validation_output_mse is not None:
                raw_record["validation_output_mse"] = record.validation_output_mse
            records.append(raw_record)
        return {
            "schema": self.schema,
            "asset_status": self.asset_status,
            "model_name": self.model_name,
            "model_num_layers": self.model_num_layers,
            "num_query_heads": self.num_query_heads,
            "head_dim": self.head_dim,
            "sequence_length": self.sequence_length,
            "method": self.method,
            "parameterization": self.parameterization,
            "objective": self.objective,
            "epsilon": self.epsilon,
            "layer_ids_zero_based": list(self.layer_ids_zero_based),
            "head_record_count": self.head_record_count,
            "heads": records,
            "provenance": _json_copy(dict(self.provenance)),
        }


@dataclass(frozen=True)
class LegacyAdaptiveGroupHeadRecord:
    """One legacy layer-padded record with the inactive tail kept only as evidence."""

    layer_id: int
    head_id: int
    group_count: int
    stored_gmax: int
    groups: tuple[tuple[int, ...], ...]
    stored_packed_lower_triangular: tuple[float, ...]
    active_packed_lower_triangular: tuple[float, ...]
    expanded_quadratic_matrix: np.ndarray
    validation_output_mse: float


@dataclass(frozen=True)
class LegacyAdaptiveGroupAsset:
    """Validated N=4096 legacy transfer asset; never a formal main-line asset."""

    schema: str
    model_name: str
    sequence_length: int
    head_dim: int
    epsilon: float
    method: str
    parameterization: str
    objective: str
    heads: Mapping[tuple[int, int], LegacyAdaptiveGroupHeadRecord]
    layer_stored_gmax: Mapping[int, int]


@dataclass(frozen=True)
class LegacyAssetMigration:
    """Pure conversion result; callers must explicitly persist it immutably."""

    canonical: CanonicalAdaptiveGroupAsset
    report: Mapping[str, object]


def expand_packed_lower_triangular(
    packed_lower_triangular: Sequence[float], group_count: int
) -> np.ndarray:
    """Reconstruct ``L @ L.T`` from torch lower-triangle row-major packing."""
    if type(group_count) is not int or group_count < 1:
        raise ValueError("group_count must be a positive integer")
    expected_length = _active_packed_length(group_count)
    if len(packed_lower_triangular) != expected_length:
        raise ValueError(
            f"packed_lower_triangular length does not match its group_count: expected {expected_length}, got {len(packed_lower_triangular)}"
        )
    lower = np.zeros((group_count + 1, group_count + 1), dtype=np.float64)
    offset = 0
    for row in range(group_count + 1):
        for column in range(row + 1):
            value = _require_finite_group_number(
                packed_lower_triangular[offset], "packed_lower_triangular"
            )
            lower[row, column] = value
            offset += 1
    return lower @ lower.T


def _expand_legacy_active_quadratic_matrix(
    active_packed_lower_triangular: Sequence[float],
    *,
    group_count: int,
    stored_gmax: int,
) -> np.ndarray:
    """Rebuild the legacy masked matrix, preserving zero inactive rows/columns."""
    active = expand_packed_lower_triangular(active_packed_lower_triangular, group_count)
    expanded = np.zeros((stored_gmax + 1, stored_gmax + 1), dtype=np.float64)
    active_dimension = group_count + 1
    expanded[:active_dimension, :active_dimension] = active
    return expanded


def _require_exact_index_keys(
    values: Mapping[str, object], count: int, name: str
) -> None:
    expected = {str(index) for index in range(count)}
    if set(values) != expected:
        raise ValueError(
            f"group asset {name} must contain exactly zero-based indices 0..{count - 1}"
        )


def _require_allowed_group_keys(
    value: Mapping[str, object], allowed: set[str], path: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(
            f"group asset {path} has unknown field(s): {', '.join(unknown)}"
        )
    missing = sorted(allowed - set(value))
    if missing:
        raise ValueError(
            f"group asset {path} is missing required field(s): {', '.join(missing)}"
        )


def _require_nonempty_group_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"group asset {path} must be a non-empty string")
    return value


def _require_finite_group_number(
    value: object,
    path: str,
    *,
    strictly_positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"group asset {path} must be a finite number")
    numeric_value = float(value)
    if strictly_positive and numeric_value <= 0:
        raise ValueError(f"group asset {path} must be positive")
    if nonnegative and numeric_value < 0:
        raise ValueError(f"group asset {path} must be non-negative")
    return numeric_value


def _require_exact_integer_list(value: object, expected: list[int], path: str) -> None:
    if not isinstance(value, list) or any((type(item) is not int for item in value)):
        raise ValueError(f"group asset {path} must be an integer list")
    if value != expected:
        raise ValueError(f"group asset {path} must equal {expected!r}")


def _validate_group_indices(value: object, path: str) -> set[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"group asset {path} must be a non-empty integer list")
    if any((type(item) is not int for item in value)):
        raise ValueError(f"group asset {path} must be an integer list")
    if any((item < 0 or item >= GROUP_ASSET_SEQUENCE_LENGTH for item in value)):
        raise ValueError(
            f"group asset {path} entries must lie in 0..{GROUP_ASSET_SEQUENCE_LENGTH - 1}"
        )
    result = set(value)
    if len(result) != len(value):
        raise ValueError(f"group asset {path} entries must be unique")
    return result


def _validate_adaptive_partition(
    value: object, group_count: int, path: str
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list) or len(value) != group_count:
        raise ValueError(f"{path} must contain exactly group_count groups")
    groups: list[tuple[int, ...]] = []
    dimensions: set[int] = set()
    for group_id, raw_group in enumerate(value):
        group_path = f"{path}[{group_id}]"
        if not isinstance(raw_group, list) or not raw_group:
            raise ValueError(f"{group_path} must be a non-empty integer list")
        if any(
            (
                type(item) is not int or item < 0 or item >= GROUP_ASSET_HEAD_DIM
                for item in raw_group
            )
        ):
            raise ValueError(f"{group_path} must contain integer dimensions in 0..63")
        if raw_group != sorted(raw_group) or len(set(raw_group)) != len(raw_group):
            raise ValueError(f"{group_path} must be strictly sorted and unique")
        overlap = dimensions & set(raw_group)
        if overlap:
            raise ValueError(
                f"{path} partition has duplicate dimensions: {sorted(overlap)!r}"
            )
        dimensions.update(raw_group)
        groups.append(tuple(raw_group))
    if dimensions != set(range(GROUP_ASSET_HEAD_DIM)):
        missing = sorted(set(range(GROUP_ASSET_HEAD_DIM)) - dimensions)
        raise ValueError(
            f"{path} partition must cover dimensions 0..63 exactly; missing={missing!r}"
        )
    return tuple(groups)


def _validate_active_packed_factor(
    value: object, group_count: int, path: str
) -> tuple[float, ...]:
    expected_length = _active_packed_length(group_count)
    if not isinstance(value, list) or len(value) != expected_length:
        raise ValueError(f"{path} must have exact active length {expected_length}")
    return tuple(
        (
            _require_finite_group_number(item, f"{path}[{index}]")
            for (index, item) in enumerate(value)
        )
    )


def _validate_expanded_quadratic_matrix(
    value: object, stored_gmax: int, path: str
) -> np.ndarray:
    dimension = stored_gmax + 1
    if not isinstance(value, list) or len(value) != dimension:
        raise ValueError(f"{path} must be a {dimension}x{dimension} matrix")
    rows: list[list[float]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != dimension:
            raise ValueError(f"{path}[{row_index}] must have length {dimension}")
        rows.append(
            [
                _require_finite_group_number(
                    cell, f"{path}[{row_index}][{column_index}]"
                )
                for (column_index, cell) in enumerate(row)
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def _stable_gap_candidate_cut_indices() -> tuple[int, ...]:
    first_tail_dimension = math.ceil(
        GROUP_ASSET_HEAD_DIM * (1.0 - STABLE_GAP_TOP_FRACTION)
    )
    return tuple(range(first_tail_dimension, GROUP_ASSET_HEAD_DIM - 1))


def _validate_finite_number_list(
    value: object, expected_length: int, path: str
) -> list[float]:
    if not isinstance(value, list) or len(value) != expected_length:
        raise ValueError(
            f"{path} must contain exactly {expected_length} finite numbers"
        )
    return [
        _require_finite_group_number(item, f"{path}[{index}]")
        for (index, item) in enumerate(value)
    ]


def _require_exact_identity_fields(
    value: Mapping[str, object], expected: Mapping[str, str | None], path: str
) -> None:
    for key, exact_value in expected.items():
        child = value.get(key)
        if exact_value is not None:
            if child != exact_value:
                raise ValueError(f"{path}.{key} must be {exact_value!r}")
        elif key.endswith("sha256"):
            try:
                _validate_sha256(child, f"{path}.{key}")
            except ValueError as exc:
                raise ValueError(f"{path}.{key} is required") from exc
        elif not isinstance(child, str) or not child:
            raise ValueError(f"{path}.{key} must be a non-empty string")


def _validate_canonical_asset_status(
    value: object, *, allow_nonformal_cpu_fixture: bool
) -> str:
    valid_statuses = {
        CANONICAL_ASSET_STATUS_FORMAL_PRIMARY,
        CANONICAL_ASSET_STATUS_LEGACY_TRANSFER,
        CANONICAL_ASSET_STATUS_NONFORMAL_CPU,
    }
    if not isinstance(value, str) or value not in valid_statuses:
        raise ValueError("canonical adaptive asset asset_status is unsupported")
    if value == CANONICAL_ASSET_STATUS_NONFORMAL_CPU and (
        not allow_nonformal_cpu_fixture
    ):
        raise ValueError(
            "nonformal CPU fixture cannot be loaded through the formal canonical asset path"
        )
    return value


def _require_canonical_adaptive_sequence_length(
    value: object, asset_status: str
) -> int:
    if asset_status == CANONICAL_ASSET_STATUS_FORMAL_PRIMARY:
        if type(value) is not int or value <= 0:
            raise ValueError(
                "canonical adaptive asset sequence_length must be a positive integer for formal_primary"
            )
        return value
    expected_length = {
        CANONICAL_ASSET_STATUS_LEGACY_TRANSFER: GROUP_ASSET_SEQUENCE_LENGTH,
        CANONICAL_ASSET_STATUS_NONFORMAL_CPU: NONFORMAL_CPU_HARNESS_SEQUENCE_LENGTH,
    }[asset_status]
    if type(value) is not int or value != expected_length:
        raise ValueError(
            f"canonical adaptive asset sequence_length must be {expected_length} for {asset_status}"
        )
    return expected_length


def _normalize_adaptive_layer_ids(
    value: Iterable[int] | object, name: str
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an explicit integer list")
    try:
        layers = list(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an explicit integer list") from exc
    if not layers:
        raise ValueError(f"{name} must not be empty")
    if any((type(layer_id) is not int for layer_id in layers)):
        raise ValueError(f"{name} must contain only integer layer IDs")
    if any((layer_id < 0 or layer_id >= GROUP_ASSET_LAYERS for layer_id in layers)):
        raise ValueError(f"{name} must lie in 0..{GROUP_ASSET_LAYERS - 1}")
    if layers != sorted(layers) or len(set(layers)) != len(layers):
        raise ValueError(f"{name} must be strictly increasing without duplicates")
    return tuple(layers)


def _require_adaptive_layer_id(value: object, path: str) -> int:
    if type(value) is not int or value < 0 or value >= GROUP_ASSET_LAYERS:
        raise ValueError(f"{path} must be an integer layer ID in 0..23")
    return value


def _require_adaptive_group_count(value: object, path: str) -> int:
    if type(value) is not int or value not in ADAPTIVE_GROUP_COUNT_RANGE:
        raise ValueError(f"{path} must be an integer group count in 2..5")
    return value


def _require_adaptive_sequence_length(value: object, path: str) -> int:
    if type(value) is not int or value not in (
        CANONICAL_ADAPTIVE_SEQUENCE_LENGTH,
        GROUP_ASSET_SEQUENCE_LENGTH,
    ):
        raise ValueError(f"{path} must be 2048 or 4096")
    return value


def _require_exact_adaptive_integer(path: str, value: object, expected: int) -> None:
    if value != expected or type(value) is not int:
        raise ValueError(f"{path} must be exactly {expected}")


def _active_packed_length(group_count: int) -> int:
    return (group_count + 1) * (group_count + 2) // 2


def _infer_group_count_from_packed_length(length: int) -> int:
    for group_count in range(1, ADAPTIVE_MAX_STORED_GMAX + 1):
        if _active_packed_length(group_count) == length:
            return group_count
    raise ValueError(
        "packed_lower_triangular length is not a triangular group factor length"
    )


def _require_unique_nonempty_strings(value: object, path: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any((not isinstance(item, str) or not item for item in value))
    ):
        raise ValueError(f"{path} must be a non-empty string list")
    if len(set(value)) != len(value):
        raise ValueError(f"{path} must be unique")
    return list(value)


def _json_copy(value: object) -> object:
    try:
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ValueError("adaptive asset value must be JSON-safe and finite") from exc
