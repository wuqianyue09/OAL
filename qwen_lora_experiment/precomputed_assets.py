"""Load released Qwen grouping/coefficient results without rerunning calibration."""

from dataclasses import dataclass
from types import MappingProxyType
from .config import PilotConfig
from .grouped_initialization import (
    GroupedHeadParameters,
    GroupedParameterInitialization,
)
from .legacy_group_assets import load_legacy_adaptive_group_asset


@dataclass(frozen=True)
class PrecomputedGroupAsset:
    initialization: GroupedParameterInitialization
    model_name: str
    sequence_length: int


def load_precomputed_group_asset(config: PilotConfig) -> PrecomputedGroupAsset:
    """Select configured heads and discard only inactive factor padding."""
    config.validate()
    source = load_legacy_adaptive_group_asset(
        config.group_asset_path, expected_sha256=config.group_asset_sha256
    )
    heads = {
        (layer, head): GroupedHeadParameters(
            record.group_count, record.groups, record.active_packed_lower_triangular
        )
        for ((layer, head), record) in source.heads.items()
        if layer in config.replacement_layer_ids
    }
    geometry = config.geometry
    initialization = GroupedParameterInitialization(
        geometry.num_layers,
        geometry.num_query_heads,
        source.head_dim,
        source.epsilon,
        config.replacement_layer_ids,
        MappingProxyType(heads),
    )
    return PrecomputedGroupAsset(
        initialization, source.model_name, source.sequence_length
    )
