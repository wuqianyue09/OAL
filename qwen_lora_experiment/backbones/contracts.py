"""Validated runtime boundary shared by supported model families."""

from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from torch import Tensor, nn
from ..attention.common import (
    AttentionContractError,
    AttentionFamilyContract,
    Qwen2AttentionContract,
)
from .spec import ModelGeometry


@dataclass(frozen=True)
class ValidatedBackbone:
    geometry: ModelGeometry
    layers: nn.ModuleList
    attention_type: type[object]
    apply_rotary_pos_emb: Callable[[Tensor, Tensor, Tensor, Tensor], object]
    transformers_version: str
    projection_shapes: tuple[tuple[int, int], ...]
    family_contract: AttentionFamilyContract
    compatibility_contract: Qwen2AttentionContract | None = None


def require_sdpa_selected(model: nn.Module) -> None:
    """Require the native backend shared by both supported model profiles."""
    config = getattr(model, "config", None)
    if getattr(config, "_attn_implementation", None) != "sdpa":
        raise AttentionContractError(
            "loaded Qwen config._attn_implementation must remain 'sdpa'"
        )
