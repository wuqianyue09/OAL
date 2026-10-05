"""Historical grouped formats and their explicit conversion to canonical assets."""

from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
import numpy as np
from .asset_io import _load_json_asset, _require_exact_keys
from .grouped_asset_contract import (
    ADAPTIVE_MAX_STORED_GMAX,
    ADAPTIVE_MULTIGROUP_METHOD,
    ADAPTIVE_QUADRATIC_PARAMETERIZATION,
    GROUP_ASSET_HEAD_DIM,
    GROUP_ASSET_LAYERS,
    GROUP_ASSET_MODEL_NAME,
    GROUP_ASSET_QUERY_HEADS,
    GROUP_ASSET_SEQUENCE_LENGTH,
    LEGACY_ADAPTIVE_QUADRATIC_OBJECTIVE,
    LEGACY_ADAPTIVE_SCHEMA,
    LegacyAdaptiveGroupAsset,
    LegacyAdaptiveGroupHeadRecord,
    _LEGACY_ADAPTIVE_HEAD_OPTIONAL_KEYS,
    _LEGACY_ADAPTIVE_HEAD_REQUIRED_KEYS,
    _LEGACY_ADAPTIVE_ROOT_OPTIONAL_KEYS,
    _LEGACY_ADAPTIVE_ROOT_REQUIRED_KEYS,
    _active_packed_length,
    _expand_legacy_active_quadratic_matrix,
    _normalize_adaptive_layer_ids,
    _require_adaptive_group_count,
    _require_exact_adaptive_integer,
    _require_exact_index_keys,
    _require_finite_group_number,
    _validate_adaptive_partition,
    _validate_expanded_quadratic_matrix,
    _validate_group_indices,
)


def load_legacy_adaptive_group_asset(
    path: str | Path, *, expected_sha256: str | None = None
) -> LegacyAdaptiveGroupAsset:
    """Load the explicit layer-padded N=4096 transfer-only legacy format."""
    raw = _load_json_asset(
        path, expected_sha256=expected_sha256, asset_name="legacy adaptive"
    )
    return validate_legacy_group_asset(raw)


def validate_legacy_group_asset(
    asset: Mapping[str, object],
) -> LegacyAdaptiveGroupAsset:
    """Validate the old full-model, layer-padded factor representation.

    The parser accepts historical files that predate an explicit ``schema``
    field, but normalizes them to ``legacy_v1_layer_padded``.  This is a
    compatibility exception only for the known N=4096 transfer asset, not a
    fallback for canonical primary assets.
    """
    if not isinstance(asset, Mapping):
        raise TypeError("legacy adaptive asset must be a mapping")
    _require_exact_keys(
        asset,
        _LEGACY_ADAPTIVE_ROOT_REQUIRED_KEYS | _LEGACY_ADAPTIVE_ROOT_OPTIONAL_KEYS,
        "legacy adaptive asset",
        required=_LEGACY_ADAPTIVE_ROOT_REQUIRED_KEYS,
    )
    if "schema" in asset and asset["schema"] != LEGACY_ADAPTIVE_SCHEMA:
        raise ValueError(
            f"legacy adaptive asset schema must be {LEGACY_ADAPTIVE_SCHEMA!r}"
        )
    if asset["model_name"] != GROUP_ASSET_MODEL_NAME:
        raise ValueError(
            f"legacy adaptive asset model_name must be {GROUP_ASSET_MODEL_NAME!r}"
        )
    _require_exact_adaptive_integer(
        "legacy adaptive head_dim", asset["head_dim"], GROUP_ASSET_HEAD_DIM
    )
    _require_exact_adaptive_integer(
        "legacy adaptive seq_len", asset["seq_len"], GROUP_ASSET_SEQUENCE_LENGTH
    )
    if asset["method"] != ADAPTIVE_MULTIGROUP_METHOD:
        raise ValueError(
            f"legacy adaptive asset method must be {ADAPTIVE_MULTIGROUP_METHOD!r}"
        )
    if asset["parameterization"] != ADAPTIVE_QUADRATIC_PARAMETERIZATION:
        raise ValueError(
            "legacy adaptive asset parameterization does not match protocol"
        )
    if asset["objective"] != LEGACY_ADAPTIVE_QUADRATIC_OBJECTIVE:
        raise ValueError("legacy adaptive asset objective does not match protocol")
    epsilon = _require_finite_group_number(
        asset["epsilon"], "legacy adaptive epsilon", strictly_positive=True
    )
    if "minimum_group_count" in asset:
        _require_exact_adaptive_integer(
            "legacy adaptive minimum_group_count", asset["minimum_group_count"], 2
        )
    declared_layers = _normalize_adaptive_layer_ids(
        asset["layer_ids_zero_based"], "legacy adaptive layer_ids_zero_based"
    )
    if declared_layers != tuple(range(GROUP_ASSET_LAYERS)):
        raise ValueError("legacy adaptive asset must cover all 24 model layers")
    index_sets = {
        name: _validate_group_indices(asset[name], f"legacy adaptive {name}")
        for name in ("train_indices", "validation_indices", "test_indices")
    }
    if any(
        (
            index_sets[left] & index_sets[right]
            for left in index_sets
            for right in index_sets
            if left < right
        )
    ):
        raise ValueError(
            "legacy adaptive asset train/validation/test indices must not overlap"
        )
    raw_layers = asset["layers"]
    if not isinstance(raw_layers, Mapping):
        raise ValueError("legacy adaptive asset layers must be an object")
    _require_exact_index_keys(raw_layers, GROUP_ASSET_LAYERS, "legacy adaptive layers")
    records: dict[tuple[int, int], LegacyAdaptiveGroupHeadRecord] = {}
    layer_stored_gmax: dict[int, int] = {}
    for layer_id in range(GROUP_ASSET_LAYERS):
        raw_layer = raw_layers[str(layer_id)]
        layer_path = f"legacy adaptive asset layers.{layer_id}"
        if not isinstance(raw_layer, Mapping):
            raise ValueError(f"{layer_path} must be an object")
        _require_exact_keys(raw_layer, {"max_groups", "heads"}, layer_path)
        stored_gmax = raw_layer["max_groups"]
        if (
            type(stored_gmax) is not int
            or stored_gmax < 2
            or stored_gmax > ADAPTIVE_MAX_STORED_GMAX
        ):
            raise ValueError(f"{layer_path}.max_groups must be an integer in 2..8")
        layer_stored_gmax[layer_id] = stored_gmax
        raw_heads = raw_layer["heads"]
        if not isinstance(raw_heads, Mapping):
            raise ValueError(f"{layer_path}.heads must be an object")
        _require_exact_index_keys(
            raw_heads, GROUP_ASSET_QUERY_HEADS, f"{layer_path}.heads"
        )
        for head_id in range(GROUP_ASSET_QUERY_HEADS):
            records[layer_id, head_id] = _validate_legacy_adaptive_head(
                raw_heads[str(head_id)],
                layer_id=layer_id,
                head_id=head_id,
                stored_gmax=stored_gmax,
                path=f"{layer_path}.heads.{head_id}",
            )
    if len(records) != GROUP_ASSET_LAYERS * GROUP_ASSET_QUERY_HEADS:
        raise ValueError(
            "legacy adaptive asset must contain exactly 336 unique head records"
        )
    return LegacyAdaptiveGroupAsset(
        schema=LEGACY_ADAPTIVE_SCHEMA,
        model_name=GROUP_ASSET_MODEL_NAME,
        sequence_length=GROUP_ASSET_SEQUENCE_LENGTH,
        head_dim=GROUP_ASSET_HEAD_DIM,
        epsilon=epsilon,
        method=ADAPTIVE_MULTIGROUP_METHOD,
        parameterization=ADAPTIVE_QUADRATIC_PARAMETERIZATION,
        objective=LEGACY_ADAPTIVE_QUADRATIC_OBJECTIVE,
        heads=MappingProxyType(records),
        layer_stored_gmax=MappingProxyType(layer_stored_gmax),
    )


def _validate_legacy_adaptive_head(
    raw_record: object, *, layer_id: int, head_id: int, stored_gmax: int, path: str
) -> LegacyAdaptiveGroupHeadRecord:
    if not isinstance(raw_record, Mapping):
        raise ValueError(f"{path} must be an object")
    allowed = _LEGACY_ADAPTIVE_HEAD_REQUIRED_KEYS | _LEGACY_ADAPTIVE_HEAD_OPTIONAL_KEYS
    _require_exact_keys(
        raw_record, allowed, path, required=_LEGACY_ADAPTIVE_HEAD_REQUIRED_KEYS
    )
    group_count = _require_adaptive_group_count(
        raw_record["group_count"], f"{path}.group_count"
    )
    if group_count > stored_gmax:
        raise ValueError(f"{path}.group_count must not exceed layer max_groups")
    groups = _validate_adaptive_partition(
        raw_record["groups"], group_count, f"{path}.groups"
    )
    stored_length = _active_packed_length(stored_gmax)
    raw_factor = raw_record["packed_lower_triangular"]
    if not isinstance(raw_factor, list) or len(raw_factor) != stored_length:
        raise ValueError(
            f"{path}.packed_lower_triangular must have the layer-padded length {stored_length}"
        )
    stored_factor = tuple(
        (
            _require_finite_group_number(
                value, f"{path}.packed_lower_triangular[{index}]"
            )
            for (index, value) in enumerate(raw_factor)
        )
    )
    active_length = _active_packed_length(group_count)
    active_factor = stored_factor[:active_length]
    expected_matrix = _expand_legacy_active_quadratic_matrix(
        active_factor, group_count=group_count, stored_gmax=stored_gmax
    )
    raw_matrix = raw_record["expanded_quadratic_matrix"]
    matrix = _validate_expanded_quadratic_matrix(
        raw_matrix, stored_gmax, f"{path}.expanded_quadratic_matrix"
    )
    if not np.allclose(matrix, expected_matrix, rtol=2e-05, atol=2e-05):
        raise ValueError(
            f"{path}.expanded_quadratic_matrix does not reconstruct from the stored factor"
        )
    if "special_dims" in raw_record:
        special_dims = raw_record["special_dims"]
        flattened = [dimension for group in groups[1:] for dimension in group]
        if special_dims != flattened:
            raise ValueError(
                f"{path}.special_dims must equal the nonzero groups in order"
            )
    validation_output_mse = _require_finite_group_number(
        raw_record["validation_output_mse"],
        f"{path}.validation_output_mse",
        nonnegative=True,
    )
    return LegacyAdaptiveGroupHeadRecord(
        layer_id=layer_id,
        head_id=head_id,
        group_count=group_count,
        stored_gmax=stored_gmax,
        groups=groups,
        stored_packed_lower_triangular=stored_factor,
        active_packed_lower_triangular=active_factor,
        expanded_quadratic_matrix=matrix,
        validation_output_mse=validation_output_mse,
    )
