"""Runtime parameters loaded from the released OAL groups and coefficients."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import numpy as np


@dataclass(frozen=True)
class GroupedHeadParameters:
    group_count: int
    groups: tuple[tuple[int, ...], ...]
    packed_lower_triangular: tuple[float, ...]


@dataclass(frozen=True)
class GroupedParameterInitialization:
    """Numerical initial state, without calibration claims or file provenance."""

    model_num_layers: int
    num_query_heads: int
    head_dim: int
    epsilon: float
    layer_ids_zero_based: tuple[int, ...]
    heads: Mapping[tuple[int, int], GroupedHeadParameters]


def build_grouped_dim_groups(state: GroupedParameterInitialization) -> np.ndarray:
    """Materialize selected layer/head partitions from the released initialization."""
    labels = np.empty(
        (len(state.layer_ids_zero_based), state.num_query_heads, state.head_dim),
        dtype=np.int32,
    )
    for offset, layer in enumerate(state.layer_ids_zero_based):
        for head in range(state.num_query_heads):
            for group, dimensions in enumerate(state.heads[layer, head].groups):
                labels[offset, head, list(dimensions)] = group
    return labels
