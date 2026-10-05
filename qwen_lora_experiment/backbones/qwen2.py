"""Thin Qwen2 family boundary over the shared attention shell."""

from __future__ import annotations
from collections.abc import Callable
import importlib
import inspect
from pathlib import Path
from torch import Tensor, nn
from ..attention.common import (
    AttentionContractError,
    AttentionFamilyContract,
    Qwen2AttentionContract,
    _extract_and_validate_family_projections,
    load_qwen2_attention_contract,
)
from .contracts import ValidatedBackbone, require_sdpa_selected
from .spec import validate_runtime_geometry

PROFILE = "qwen2_5_0_5b"
MODEL_TYPE = "qwen2"
ATTENTION_MODULE = "transformers.models.qwen2.modeling_qwen2"
ATTENTION_CLASS_NAME = "Qwen2Attention"
FAMILY_CONTRACT = AttentionFamilyContract("qwen2", qkv_bias=True, output_bias=False)


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
    """Apply Qwen's implementation to model-owned position embeddings."""
    return resolve_apply_rotary_pos_emb()(q, k, cos, sin)


def model_layers(model: object) -> nn.ModuleList:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise AttentionContractError("Qwen model.model.layers must be an nn.ModuleList")
    return layers


def validate_model(
    model: object,
    *,
    compatibility_path: str | Path,
    importer: Callable[[str], object] = importlib.import_module,
    supplied_apply_rotary_pos_emb: (
        Callable[[Tensor, Tensor, Tensor, Tensor], object] | None
    ) = None,
) -> ValidatedBackbone:
    """Validate native Qwen structure and runtime against the measured fixture."""
    if not isinstance(model, nn.Module):
        raise AttentionContractError("loaded Qwen model must be an nn.Module")
    contract = load_qwen2_attention_contract(compatibility_path)
    attention_type, rotary = _validate_runtime_transformers_contract(
        contract,
        importer=importer,
        supplied_apply_rotary_pos_emb=supplied_apply_rotary_pos_emb,
    )
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != MODEL_TYPE:
        raise AttentionContractError("loaded Qwen config.model_type must be 'qwen2'")
    geometry = validate_runtime_geometry(PROFILE, config)
    require_sdpa_selected(model)
    layers = model_layers(model)
    if len(layers) != geometry.num_layers:
        raise AttentionContractError(
            f"loaded Qwen model must have {geometry.num_layers} decoder layers"
        )
    for layer_id, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        if (
            not isinstance(attention, nn.Module)
            or type(attention) is not attention_type
        ):
            raise AttentionContractError(
                f"layer {layer_id} attention class must be the runtime {contract.attention_module}.{contract.attention_class_name} identity"
            )
        try:
            signature = str(inspect.signature(attention.forward))
        except (TypeError, ValueError) as exc:
            raise AttentionContractError(
                f"layer {layer_id} Qwen attention forward signature is not inspectable"
            ) from exc
        if signature != contract.attention_forward_signature:
            raise AttentionContractError(
                f"layer {layer_id} Qwen attention forward signature does not match fixture"
            )
        _extract_and_validate_family_projections(attention, geometry, FAMILY_CONTRACT)
    return ValidatedBackbone(
        geometry=geometry,
        layers=layers,
        attention_type=attention_type,
        apply_rotary_pos_emb=rotary,
        transformers_version=contract.transformers_version,
        projection_shapes=(
            contract.q_projection_shape,
            contract.k_projection_shape,
            contract.v_projection_shape,
            contract.o_projection_shape,
        ),
        family_contract=FAMILY_CONTRACT,
        compatibility_contract=contract,
    )


def _validate_runtime_transformers_contract(
    contract: Qwen2AttentionContract,
    *,
    importer: Callable[[str], object],
    supplied_apply_rotary_pos_emb: (
        Callable[[Tensor, Tensor, Tensor, Tensor], object] | None
    ),
) -> tuple[type[object], Callable[[Tensor, Tensor, Tensor, Tensor], object]]:
    """Prove the loaded process exposes exactly the fixture's Qwen API."""
    try:
        transformers = importer("transformers")
    except Exception as exc:
        raise AttentionContractError(
            "could not import runtime Transformers for fixture validation"
        ) from exc
    version = getattr(transformers, "__version__", None)
    if version != contract.transformers_version:
        raise AttentionContractError(
            f"runtime Transformers version does not match fixture: expected {contract.transformers_version!r}, got {version!r}"
        )
    try:
        modeling = importer(contract.attention_module)
    except Exception as exc:
        raise AttentionContractError(
            f"could not import fixture Qwen modeling module {contract.attention_module!r}"
        ) from exc
    attention_class = getattr(modeling, contract.attention_class_name, None)
    if not isinstance(attention_class, type):
        raise AttentionContractError(
            f"fixture modeling module has no class {contract.attention_class_name!r}"
        )
    rotary = getattr(modeling, "apply_rotary_pos_emb", None)
    if not callable(rotary):
        raise AttentionContractError(
            "fixture Qwen modeling module has no callable apply_rotary_pos_emb"
        )
    try:
        rotary_signature = str(inspect.signature(rotary))
    except (TypeError, ValueError) as exc:
        raise AttentionContractError(
            "runtime RoPE signature is not inspectable"
        ) from exc
    if rotary_signature != contract.rotary_signature:
        raise AttentionContractError(
            f"runtime RoPE signature does not match fixture: expected {contract.rotary_signature!r}, got {rotary_signature!r}"
        )
    if (
        supplied_apply_rotary_pos_emb is not None
        and supplied_apply_rotary_pos_emb is not rotary
    ):
        raise AttentionContractError(
            "provided apply_rotary_pos_emb must be the fixture-validated runtime callable"
        )
    return (attention_class, rotary)
