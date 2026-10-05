"""Thin Llama 3.2 family boundary over the shared attention shell."""

from __future__ import annotations
from collections.abc import Callable
import importlib
import torch
from torch import Tensor, nn
from ..attention.common import (
    AttentionContractError,
    AttentionFamilyContract,
    _extract_and_validate_family_projections,
)
from .contracts import ValidatedBackbone, require_sdpa_selected
from .spec import validate_runtime_geometry

PROFILE = "llama3_2_1b_base"
MODEL_TYPE = "llama"
ATTENTION_MODULE = "transformers.models.llama.modeling_llama"
ATTENTION_CLASS_NAME = "LlamaAttention"
FAMILY_CONTRACT = AttentionFamilyContract("llama", qkv_bias=False, output_bias=False)


def native_attention_type(
    *, importer: Callable[[str], object] = importlib.import_module
) -> type[object]:
    modeling = importer(ATTENTION_MODULE)
    attention_type = getattr(modeling, ATTENTION_CLASS_NAME, None)
    if not isinstance(attention_type, type):
        raise AttentionContractError(
            f"{ATTENTION_MODULE} has no {ATTENTION_CLASS_NAME} class"
        )
    return attention_type


def resolve_apply_rotary_pos_emb(
    *, importer: Callable[[str], object] = importlib.import_module
) -> Callable[[Tensor, Tensor, Tensor, Tensor], object]:
    modeling = importer(ATTENTION_MODULE)
    rotary = getattr(modeling, "apply_rotary_pos_emb", None)
    if not callable(rotary):
        raise AttentionContractError(f"{ATTENTION_MODULE} has no apply_rotary_pos_emb")
    return rotary


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> object:
    """Apply Llama's implementation to its model-owned (possibly scaled) cos/sin."""
    return resolve_apply_rotary_pos_emb()(q, k, cos, sin)


def model_layers(model: object) -> nn.ModuleList:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise AttentionContractError(
            "Llama model.model.layers must be an nn.ModuleList"
        )
    return layers


def validate_model(
    model: object,
    *,
    importer: Callable[[str], object] = importlib.import_module,
    supplied_apply_rotary_pos_emb: (
        Callable[[Tensor, Tensor, Tensor, Tensor], object] | None
    ) = None,
) -> ValidatedBackbone:
    """Validate actual Llama structure without copying Qwen's string fixture gate."""
    if not isinstance(model, nn.Module):
        raise AttentionContractError("loaded Llama model must be an nn.Module")
    try:
        transformers = importer("transformers")
        attention_type = native_attention_type(importer=importer)
        rotary = resolve_apply_rotary_pos_emb(importer=importer)
    except Exception as exc:
        raise AttentionContractError(
            "could not import the runtime Transformers Llama implementation"
        ) from exc
    version = getattr(transformers, "__version__", None)
    if not isinstance(version, str) or not version:
        raise AttentionContractError(
            "runtime Transformers must expose its actual version"
        )
    if (
        supplied_apply_rotary_pos_emb is not None
        and supplied_apply_rotary_pos_emb is not rotary
    ):
        raise AttentionContractError(
            "provided apply_rotary_pos_emb must be the runtime Llama callable"
        )
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != MODEL_TYPE:
        raise AttentionContractError("loaded Llama config.model_type must be 'llama'")
    geometry = validate_runtime_geometry(PROFILE, config)
    require_sdpa_selected(model)
    layers = model_layers(model)
    if len(layers) != geometry.num_layers:
        raise AttentionContractError(
            f"loaded Llama model must have {geometry.num_layers} decoder layers"
        )
    for layer_id, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        if type(attention) is not attention_type:
            raise AttentionContractError(
                f"layer {layer_id} self_attn must be the native {ATTENTION_CLASS_NAME}"
            )
        _extract_and_validate_family_projections(attention, geometry, FAMILY_CONTRACT)
    _validate_limited_native_forward(model, layers[0].self_attn, geometry.hidden_size)
    shapes = (
        (geometry.hidden_size, geometry.hidden_size),
        (geometry.num_kv_heads * geometry.head_dim, geometry.hidden_size),
        (geometry.num_kv_heads * geometry.head_dim, geometry.hidden_size),
        (geometry.hidden_size, geometry.hidden_size),
    )
    return ValidatedBackbone(
        geometry, layers, attention_type, rotary, version, shapes, FAMILY_CONTRACT
    )


def _validate_limited_native_forward(
    model: nn.Module, attention: nn.Module, hidden_size: int
) -> None:
    """Exercise the live two-value Llama attention API without changing weights."""
    rotary = getattr(getattr(model, "model", None), "rotary_emb", None)
    if not callable(rotary):
        raise AttentionContractError(
            "loaded Llama model.model.rotary_emb must provide model-owned cos/sin"
        )
    q_weight = getattr(getattr(attention, "q_proj", None), "weight", None)
    if not isinstance(q_weight, Tensor):
        raise AttentionContractError(
            "loaded Llama attention q_proj weight is unavailable"
        )
    sequence_length = 2
    hidden_states = torch.zeros(
        (1, sequence_length, hidden_size), device=q_weight.device, dtype=q_weight.dtype
    )
    cache_position = torch.arange(sequence_length, device=q_weight.device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask = torch.triu(
        torch.full(
            (1, 1, sequence_length, sequence_length),
            torch.finfo(q_weight.dtype).min,
            device=q_weight.device,
            dtype=q_weight.dtype,
        ),
        diagonal=1,
    )
    was_training = attention.training
    try:
        attention.eval()
        with torch.no_grad():
            position_embeddings = rotary(hidden_states, position_ids)
            result = attention(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                past_key_value=None,
                cache_position=cache_position,
            )
    except Exception as exc:
        raise AttentionContractError(
            "limited native Llama attention forward failed"
        ) from exc
    finally:
        attention.train(was_training)
    if not isinstance(result, tuple) or len(result) != 2:
        raise AttentionContractError(
            "native Llama attention forward must return exactly two values"
        )
    output, attention_weights = result
    if not isinstance(output, Tensor) or tuple(output.shape) != (
        1,
        sequence_length,
        hidden_size,
    ):
        raise AttentionContractError(
            "native Llama attention forward returned an unexpected output shape"
        )
    if attention_weights is not None and (not isinstance(attention_weights, Tensor)):
        raise AttentionContractError(
            "native Llama attention weights must be a tensor or None"
        )
