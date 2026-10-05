"""Grouped asset identity and live model/data provenance."""

from __future__ import annotations
from collections.abc import Mapping
from .config import PilotConfig
from .paths import sha256_file


def _grouped_base_asset_runtime_identity(
    config: PilotConfig,
    *,
    model_identity: Mapping[str, object],
    data_manifest: Mapping[str, object] | None,
    validate_live_provenance: bool = True,
) -> tuple[object, dict[str, object]]:
    """Load the shared Grouped base asset and optional composite stats identity."""
    from .precomputed_assets import load_precomputed_group_asset

    asset = load_precomputed_group_asset(config)
    state = asset.initialization
    counts = [record.group_count for record in state.heads.values()]
    return (
        asset,
        {
            "adaptive_multigroup_asset": {
                "schema": "legacy_v1_layer_padded",
                "asset_sha256": sha256_file(config.group_asset_path),
                "declared_sha256": config.group_asset_sha256,
                "asset_status": "precomputed",
                "method": "adaptive_multigroup_full_quadratic",
                "sequence_length": asset.sequence_length,
                "model_num_layers": state.model_num_layers,
                "layer_ids_zero_based": list(state.layer_ids_zero_based),
                "head_record_count": len(state.heads),
            },
            "group_count_histogram": {
                str(count): counts.count(count) for count in sorted(set(counts))
            },
            "maximum_group_count": max(counts),
            "padded_gmax": 8,
            "factor_initialization_source": "precomputed_asset",
        },
    )
