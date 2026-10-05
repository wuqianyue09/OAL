"""Dependency-free model geometry and experiment scope definitions."""

from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping, cast

ModelProfile = Literal["qwen2_5_0_5b", "llama3_2_1b_base"]


@dataclass(frozen=True)
class ModelGeometry:
    num_layers: int
    hidden_size: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int


@dataclass(frozen=True)
class ExperimentScope:
    replacement_layers: tuple[int, ...]
    lora_layers: tuple[int, ...]

    def actual_replacement_layers(self, method: str) -> tuple[int, ...]:
        return self.replacement_layers


_PROFILE_GEOMETRIES: Mapping[ModelProfile, ModelGeometry] = MappingProxyType(
    {
        "qwen2_5_0_5b": ModelGeometry(24, 896, 14, 2, 64),
        "llama3_2_1b_base": ModelGeometry(16, 2048, 32, 8, 64),
    }
)


def geometry_for_profile(model_profile: str) -> ModelGeometry:
    """Return the immutable expected geometry for one supported profile."""
    try:
        return _PROFILE_GEOMETRIES[cast(ModelProfile, model_profile)]
    except KeyError as exc:
        raise ValueError(
            "model_profile must be 'qwen2_5_0_5b' or 'llama3_2_1b_base'"
        ) from exc


def validate_runtime_geometry(
    model_profile: str, actual_config: object
) -> ModelGeometry:
    """Validate an actual Hugging Face config and return its instance geometry."""
    field_sources = {
        "num_layers": "num_hidden_layers",
        "hidden_size": "hidden_size",
        "num_query_heads": "num_attention_heads",
        "num_kv_heads": "num_key_value_heads",
        "head_dim": "head_dim",
    }
    values: dict[str, int] = {}
    for field_name, source_name in field_sources.items():
        value = getattr(actual_config, source_name, None)
        if field_name == "head_dim" and value is None:
            hidden_size = values.get("hidden_size")
            num_query_heads = values.get("num_query_heads")
            if (
                hidden_size is not None
                and num_query_heads
                and (hidden_size % num_query_heads == 0)
            ):
                value = hidden_size // num_query_heads
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"runtime model config must provide integer {source_name}")
        values[field_name] = value
    actual = ModelGeometry(**values)
    expected = geometry_for_profile(model_profile)
    for field_name in field_sources:
        if getattr(actual, field_name) != getattr(expected, field_name):
            raise ValueError(
                f"runtime {field_name} does not match model_profile {model_profile!r}: expected {getattr(expected, field_name)}, got {getattr(actual, field_name)}"
            )
    return actual


def make_scope(
    geometry: ModelGeometry, replacement_layers: tuple[int, ...]
) -> ExperimentScope:
    """Build the requested replacement scope and fixed all-layer LoRA scope."""
    return ExperimentScope(replacement_layers, tuple(range(geometry.num_layers)))
