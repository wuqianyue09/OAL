"""Dependency-free value primitives and selector contracts for HD Block-GEMM."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any

_MAX_HEAD_DIMENSION = 64


def _stable_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _require_plain_int(value: object, *, name: str, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        comparison = "positive" if minimum == 1 else f">= {minimum}"
        raise ValueError(f"{name} must be a {comparison} integer")
    return value


def _require_head_dimension(value: object) -> int:
    head_dimension = _require_plain_int(value, name="head_dimension", minimum=1)
    if head_dimension > _MAX_HEAD_DIMENSION:
        raise ValueError(
            f"head_dimension must be no greater than {_MAX_HEAD_DIMENSION}"
        )
    return head_dimension


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    reversed_strides: list[int] = []
    for dimension in reversed(shape):
        reversed_strides.append(stride)
        stride *= dimension
    return tuple(reversed(reversed_strides))


def _validate_feature_padding(value: object, *, head_dimension: int) -> str:
    """Canonicalize the private logical-to-physical feature layout selector."""
    if not isinstance(value, str):
        raise TypeError("feature_padding must be a string")
    if value not in ("none", "f2160"):
        raise ValueError("feature_padding must be 'none' or 'f2160'")
    if value == "f2160" and head_dimension != 64:
        raise ValueError("feature_padding='f2160' requires D=64")
    return value


_QUERY_OPERATOR_CONFIG_SCHEMA = "hd_query_operator_config_v4"
_QUERY_FEATURE_IMPLS = frozenset(("generic_materialized", "triton_materialized"))
_KEY_FEATURE_IMPLS = frozenset(("generic_materialized", "triton_materialized"))
_KEY_FOLD_IMPLS = frozenset(("generic_materialized", "triton_materialized"))
_QUERY_FOLD_IMPLS = frozenset(("generic_materialized", "triton_materialized"))
_QUERY_GRADIENT_FLOWS = frozenset(("materialized",))
_BACKWARD_SCHEDULES = frozenset(("split", "shared_wave"))
_GRADIENT_STAGINGS = frozenset(("per_wave", "full_bf16"))
_FORWARD_NORMALIZE_IMPLS = frozenset(("torch", "triton"))
_BACKWARD_NORMALIZE_IMPLS = frozenset(("torch", "triton"))
_KV_CROSS_IMPLS = frozenset(("split", "batched_dense"))
_QUERY_FEATURE_TOKEN_TILES = frozenset((1, 2, 4))
_QUERY_FOLD_TOKEN_TILES = frozenset((1, 2, 4))
_QUERY_FOLD_INPUTS = frozenset(("staged_fp32", "raw"))
_KEY_RETENTIONS = frozenset(("none", "forward", "backward"))
_V4_OPERATOR_CONFIG_FIELDS = frozenset(
    (
        "schema_version",
        "query_feature_impl",
        "query_fold_impl",
        "query_gradient_flow",
        "query_producer_fold_strategy",
        "query_consumer_stages",
    )
)
_LEGACY_OPERATOR_CONFIG_FIELDS = frozenset(
    (
        "query_feature_impl",
        "query_fold_impl",
        "query_gradient_flow",
        "query_consumer_fusion",
    )
)

# One ordered default table for independent config extraction and omission.
_HD_OPERATOR_CONFIG_DEFAULTS = (
    ("key_feature_impl", "generic_materialized"),
    ("key_fold_impl", "generic_materialized"),
    ("feature_padding", "none"),
    ("query_feature_token_tile", 1),
    ("query_fold_token_tile", 1),
    ("query_fold_input", "staged_fp32"),
    ("backward_schedule", "split"),
    ("gradient_staging", "per_wave"),
    ("forward_normalize_impl", "torch"),
    ("backward_normalize_impl", "torch"),
    ("kv_cross_impl", "split"),
    ("save_local_score", False),
    ("key_retention", "none"),
)


def _canonical_consumer_stages(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("query_consumer_stages must be a sequence")
    if tuple(value):
        raise ValueError("query_consumer_stages: OP-4 exploration is retired")
    return ()


def _canonical_v4_config(
    *,
    query_feature_impl: object,
    query_fold_impl: object,
    query_gradient_flow: object,
    query_producer_fold_strategy: object,
    query_consumer_stages: object,
) -> dict[str, object]:
    for name, value, allowed in (
        ("query_feature_impl", query_feature_impl, _QUERY_FEATURE_IMPLS),
        ("query_fold_impl", query_fold_impl, _QUERY_FOLD_IMPLS),
        ("query_gradient_flow", query_gradient_flow, _QUERY_GRADIENT_FLOWS),
        ("query_producer_fold_strategy", query_producer_fold_strategy, ("none",)),
    ):
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
        if value not in allowed:
            if name in ("query_gradient_flow", "query_producer_fold_strategy"):
                raise ValueError(f"{name}: OP-3 exploration is retired")
            raise ValueError(f"unsupported {name}")
    stages = _canonical_consumer_stages(query_consumer_stages)
    return {
        "schema_version": _QUERY_OPERATOR_CONFIG_SCHEMA,
        "query_feature_impl": query_feature_impl,
        "query_fold_impl": query_fold_impl,
        "query_gradient_flow": query_gradient_flow,
        "query_producer_fold_strategy": query_producer_fold_strategy,
        "query_consumer_stages": list(stages),
    }


def _validate_backward_schedule(
    backward_schedule: object,
    *,
    requested_gradient_mask: tuple[bool, bool, bool, bool, bool, bool],
    query_gradient_flow: object,
    query_consumer_stages: object,
) -> str:
    """Return the executable schedule for this exact gradient selection."""
    if not isinstance(backward_schedule, str):
        raise TypeError("backward_schedule must be a string")
    if backward_schedule not in _BACKWARD_SCHEDULES:
        raise ValueError("unsupported backward_schedule")
    if backward_schedule == "split":
        return backward_schedule
    need_q, need_k, need_v, need_a, need_b, need_c = requested_gradient_mask
    query_side = need_q or need_a or need_b or need_c
    kv_side = need_k or need_v
    if not (query_side and kv_side):
        return "split"
    if query_gradient_flow != "materialized" or tuple(query_consumer_stages):
        raise ValueError(
            "shared_wave requires materialized query gradients without OP-4 consumers"
        )
    return backward_schedule


def _validate_gradient_staging(
    gradient_staging: object,
    *,
    precision: object,
) -> str:
    """Return the executable backward gradient staging policy."""
    if not isinstance(gradient_staging, str):
        raise TypeError("gradient_staging must be a string")
    if gradient_staging not in _GRADIENT_STAGINGS:
        raise ValueError("unsupported gradient_staging")
    if gradient_staging == "full_bf16" and precision != "bf16_tensorcore":
        raise ValueError(
            "full_bf16 gradient staging requires bf16_tensorcore precision"
        )
    return gradient_staging


def _validate_forward_normalize_impl(
    forward_normalize_impl: object,
    *,
    precision: object,
) -> str:
    """Return the executable forward normalization implementation."""
    if not isinstance(forward_normalize_impl, str):
        raise TypeError("forward_normalize_impl must be a string")
    if forward_normalize_impl not in _FORWARD_NORMALIZE_IMPLS:
        raise ValueError("unsupported forward_normalize_impl")
    if forward_normalize_impl == "triton" and precision != "bf16_tensorcore":
        raise ValueError(
            "Triton forward normalization requires bf16_tensorcore precision"
        )
    return forward_normalize_impl


def _validate_backward_normalize_impl(
    backward_normalize_impl: object,
    *,
    precision: object,
) -> str:
    """Return the executable backward normalization implementation."""
    if not isinstance(backward_normalize_impl, str):
        raise TypeError("backward_normalize_impl must be a string")
    if backward_normalize_impl not in _BACKWARD_NORMALIZE_IMPLS:
        raise ValueError("unsupported backward_normalize_impl")
    if backward_normalize_impl == "triton" and precision != "bf16_tensorcore":
        raise ValueError(
            "Triton backward normalization requires bf16_tensorcore precision"
        )
    return backward_normalize_impl


def _validate_kv_cross_impl(
    kv_cross_impl: object,
    *,
    precision: object,
) -> str:
    """Return the executable KV-cross contraction implementation."""
    del precision
    if not isinstance(kv_cross_impl, str):
        raise TypeError("kv_cross_impl must be a string")
    if kv_cross_impl not in _KV_CROSS_IMPLS:
        raise ValueError("unsupported kv_cross_impl")
    return kv_cross_impl


def _validate_query_feature_token_tile(
    query_feature_token_tile: object,
    *,
    query_feature_impl: object,
) -> int:
    """Return the token count owned by one query-feature program."""
    if not isinstance(query_feature_token_tile, int) or isinstance(
        query_feature_token_tile, bool
    ):
        raise TypeError("query_feature_token_tile must be an integer")
    if query_feature_token_tile not in _QUERY_FEATURE_TOKEN_TILES:
        raise ValueError("unsupported query_feature_token_tile")
    if query_feature_token_tile != 1 and query_feature_impl != "triton_materialized":
        raise ValueError(
            "query_feature_token_tile greater than one requires triton_materialized"
        )
    return query_feature_token_tile


def _validate_query_fold_token_tile(
    query_fold_token_tile: object,
    *,
    query_fold_impl: object,
) -> int:
    """Validate the private number of independent dQ tokens per program."""
    if not isinstance(query_fold_token_tile, int) or isinstance(
        query_fold_token_tile, bool
    ):
        raise TypeError("query_fold_token_tile must be an integer")
    if query_fold_token_tile not in _QUERY_FOLD_TOKEN_TILES:
        raise ValueError("unsupported query_fold_token_tile")
    if query_fold_token_tile != 1 and query_fold_impl != "triton_materialized":
        raise ValueError(
            "query_fold_token_tile greater than one requires triton_materialized"
        )
    return query_fold_token_tile


def _validate_query_fold_input(
    query_fold_input: object,
    *,
    query_fold_impl: object,
    query_gradient_flow: object,
) -> str:
    """Validate the physical Q operand consumed by the materialized fold."""
    if not isinstance(query_fold_input, str):
        raise TypeError("query_fold_input must be a string")
    if query_fold_input not in _QUERY_FOLD_INPUTS:
        raise ValueError("unsupported query_fold_input")
    if query_fold_input == "raw" and (
        query_fold_impl != "triton_materialized"
        or query_gradient_flow != "materialized"
    ):
        raise ValueError(
            "query_fold_input=raw requires materialized triton_materialized query fold"
        )
    return query_fold_input


def _validate_save_local_score(
    save_local_score: object,
    *,
    requested_gradient_mask: tuple[bool, bool, bool, bool, bool, bool],
    precision: object,
) -> bool:
    """Return whether this exact plan can consume a saved BF16 local score."""
    if type(save_local_score) is not bool:
        raise TypeError("save_local_score must be a bool")
    if not requested_gradient_mask[2]:
        return False
    if save_local_score and precision != "bf16_tensorcore":
        raise ValueError("save_local_score requires bf16_tensorcore precision")
    return save_local_score


def _validate_key_retention(
    key_retention: object,
    *,
    precision: object,
    requested_gradient_mask: tuple[bool, bool, bool, bool, bool, bool] | None = None,
) -> str:
    """Return the executable lifetime for materialized key features."""
    if not isinstance(key_retention, str):
        raise TypeError("key_retention must be a string")
    if key_retention not in _KEY_RETENTIONS:
        raise ValueError("unsupported key_retention")
    if key_retention != "none" and precision != "bf16_tensorcore":
        raise ValueError("key retention requires bf16_tensorcore precision")
    if key_retention == "backward" and requested_gradient_mask is not None:
        need_q, _, need_v, need_a, need_b, need_c = requested_gradient_mask
        query_side = need_q or need_a or need_b or need_c
        if not (query_side or need_v):
            return "forward"
    return key_retention


def _validate_legacy_consumer_fusion(fusion: object) -> None:
    """Accept the retired selector only in its no-fusion form."""
    if not isinstance(fusion, str):
        raise TypeError("query_consumer_fusion must be a string")
    if fusion != "none":
        raise ValueError("query_consumer_fusion: OP-4 exploration is retired")


def _canonical_legacy_config(config: Mapping[str, object]) -> dict[str, object]:
    _validate_legacy_consumer_fusion(config.get("query_consumer_fusion"))
    return _canonical_v4_config(
        query_feature_impl=config.get("query_feature_impl"),
        query_fold_impl=config.get("query_fold_impl"),
        query_gradient_flow=config.get("query_gradient_flow"),
        query_producer_fold_strategy="none",
        query_consumer_stages=(),
    )


def canonicalize_query_operator_config(config: object) -> dict[str, object]:
    """Canonicalize a legacy or v4 structural query-operator config."""
    if not isinstance(config, Mapping):
        raise TypeError("operator config must be a mapping")
    payload = dict(config)
    fields = set(payload)
    if fields == _V4_OPERATOR_CONFIG_FIELDS:
        if payload.get("schema_version") != _QUERY_OPERATOR_CONFIG_SCHEMA:
            raise ValueError("operator config schema_version is invalid")
        return _canonical_v4_config(
            query_feature_impl=payload.get("query_feature_impl"),
            query_fold_impl=payload.get("query_fold_impl"),
            query_gradient_flow=payload.get("query_gradient_flow"),
            query_producer_fold_strategy=payload.get("query_producer_fold_strategy"),
            query_consumer_stages=payload.get("query_consumer_stages"),
        )
    if fields == _LEGACY_OPERATOR_CONFIG_FIELDS:
        return _canonical_legacy_config(payload)
    if fields == _V4_OPERATOR_CONFIG_FIELDS | {"query_consumer_fusion"}:
        v4 = canonicalize_query_operator_config(
            {name: payload[name] for name in _V4_OPERATOR_CONFIG_FIELDS}
        )
        try:
            _validate_legacy_consumer_fusion(payload["query_consumer_fusion"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                "mixed legacy/v4 query operator selections conflict"
            ) from error
        return v4
    raise ValueError("operator config must be an exact legacy or v4 selector structure")


def canonicalize_hd_operator_config(
    config: object,
    *,
    precision: str = "bf16_tensorcore",
    head_dimension: int = 64,
    result_contract: str = "full_aux",
) -> dict[str, object]:
    """Canonicalize query selectors plus independent private HD selectors.

    Default-valued independent selectors are omitted so historical query-only
    identities remain stable.  Validation uses the complete-gradient contract
    exercised by inventory, formal, and profile evidence.
    """
    if not isinstance(config, Mapping):
        raise TypeError("operator config must be a mapping")
    query_config = dict(config)
    independent = {
        name: query_config.pop(name, default)
        for name, default in _HD_OPERATOR_CONFIG_DEFAULTS
    }
    canonical = canonicalize_query_operator_config(query_config)
    full_gradient_mask = (True, True, True, True, True, True)
    validated = {
        "key_feature_impl": _validate_key_feature_impl(
            key_feature_impl=independent["key_feature_impl"],
            precision=precision,
            head_dimension=head_dimension,
            result_contract=result_contract,
        ),
        "key_fold_impl": _validate_key_fold_impl(
            key_fold_impl=independent["key_fold_impl"],
            precision=precision,
            head_dimension=head_dimension,
            result_contract=result_contract,
        ),
        "feature_padding": _validate_feature_padding(
            independent["feature_padding"],
            head_dimension=head_dimension,
        ),
        "query_feature_token_tile": _validate_query_feature_token_tile(
            independent["query_feature_token_tile"],
            query_feature_impl=canonical["query_feature_impl"],
        ),
        "query_fold_token_tile": _validate_query_fold_token_tile(
            independent["query_fold_token_tile"],
            query_fold_impl=canonical["query_fold_impl"],
        ),
        "query_fold_input": _validate_query_fold_input(
            independent["query_fold_input"],
            query_fold_impl=canonical["query_fold_impl"],
            query_gradient_flow=canonical["query_gradient_flow"],
        ),
        "backward_schedule": _validate_backward_schedule(
            independent["backward_schedule"],
            requested_gradient_mask=full_gradient_mask,
            query_gradient_flow=canonical["query_gradient_flow"],
            query_consumer_stages=canonical["query_consumer_stages"],
        ),
        "gradient_staging": _validate_gradient_staging(
            independent["gradient_staging"],
            precision=precision,
        ),
        "forward_normalize_impl": _validate_forward_normalize_impl(
            independent["forward_normalize_impl"],
            precision=precision,
        ),
        "backward_normalize_impl": _validate_backward_normalize_impl(
            independent["backward_normalize_impl"],
            precision=precision,
        ),
        "kv_cross_impl": _validate_kv_cross_impl(
            independent["kv_cross_impl"],
            precision=precision,
        ),
        "save_local_score": _validate_save_local_score(
            independent["save_local_score"],
            requested_gradient_mask=full_gradient_mask,
            precision=precision,
        ),
        "key_retention": _validate_key_retention(
            independent["key_retention"],
            precision=precision,
            requested_gradient_mask=full_gradient_mask,
        ),
    }
    defaults = dict(_HD_OPERATOR_CONFIG_DEFAULTS)
    canonical.update(
        {name: value for name, value in validated.items() if value != defaults[name]}
    )
    return canonical


def canonicalize_query_operator_selection(
    *,
    query_feature_impl: object,
    query_fold_impl: object,
    query_gradient_flow: object,
    query_producer_fold_strategy: object | None = None,
    query_consumer_stages: object | None = None,
    query_consumer_fusion: object | None = None,
) -> dict[str, object]:
    """Resolve current options and default-only legacy selectors once."""
    legacy_only = (
        query_consumer_fusion is not None
        and query_producer_fold_strategy is None
        and query_consumer_stages is None
    )
    # Preserve the legacy entry point's validation order for invalid inputs.
    if legacy_only:
        _validate_legacy_consumer_fusion(query_consumer_fusion)
    canonical = _canonical_v4_config(
        query_feature_impl=query_feature_impl,
        query_fold_impl=query_fold_impl,
        query_gradient_flow=query_gradient_flow,
        query_producer_fold_strategy=(
            "none"
            if query_producer_fold_strategy is None
            else query_producer_fold_strategy
        ),
        query_consumer_stages=(
            () if query_consumer_stages is None else query_consumer_stages
        ),
    )
    if query_consumer_fusion is not None and not legacy_only:
        _validate_legacy_consumer_fusion(query_consumer_fusion)
    return canonical


def _validate_query_operator_identity(
    *,
    query_feature_impl: object,
    query_fold_impl: object,
    query_gradient_flow: object,
    query_producer_fold_strategy: object | None = None,
    query_consumer_stages: object | None = None,
    query_consumer_fusion: object | None = None,
    precision: object,
    head_dimension: object,
    result_contract: object,
) -> dict[str, object]:
    """Validate and return the canonical staged private selection."""
    canonical = canonicalize_query_operator_selection(
        query_feature_impl=query_feature_impl,
        query_fold_impl=query_fold_impl,
        query_gradient_flow=query_gradient_flow,
        query_producer_fold_strategy=query_producer_fold_strategy,
        query_consumer_stages=query_consumer_stages,
        query_consumer_fusion=query_consumer_fusion,
    )
    if (
        canonical["query_feature_impl"] == "triton_materialized"
        or canonical["query_fold_impl"] == "triton_materialized"
    ) and (
        precision != "bf16_tensorcore"
        or head_dimension != 64
        or result_contract != "full_aux"
    ):
        raise ValueError(
            "Triton query identities require bf16_tensorcore, D=64, and full_aux"
        )
    return canonical


def _validate_key_feature_impl(
    *,
    key_feature_impl: object,
    precision: object,
    head_dimension: object,
    result_contract: object,
) -> str:
    """Validate the independent key-feature materialization selector."""
    if not isinstance(key_feature_impl, str):
        raise TypeError("key_feature_impl must be a string")
    if key_feature_impl not in _KEY_FEATURE_IMPLS:
        raise ValueError("unsupported key_feature_impl")
    if key_feature_impl == "triton_materialized" and (
        precision != "bf16_tensorcore"
        or head_dimension != 64
        or result_contract != "full_aux"
    ):
        raise ValueError(
            "Triton key identities require bf16_tensorcore, D=64, and full_aux"
        )
    return key_feature_impl


def _validate_key_fold_impl(
    *,
    key_fold_impl: object,
    precision: object,
    head_dimension: object,
    result_contract: object,
) -> str:
    """Validate the independent key-gradient fold selector."""
    if not isinstance(key_fold_impl, str):
        raise TypeError("key_fold_impl must be a string")
    if key_fold_impl not in _KEY_FOLD_IMPLS:
        raise ValueError("unsupported key_fold_impl")
    if key_fold_impl == "triton_materialized" and (
        precision != "bf16_tensorcore"
        or head_dimension != 64
        or result_contract != "full_aux"
    ):
        raise ValueError(
            "Triton key fold identities require bf16_tensorcore, D=64, and full_aux"
        )
    return key_fold_impl


def _require_implemented_query_operator_identity(
    *,
    query_feature_impl: object,
    query_fold_impl: object,
    query_gradient_flow: object,
    query_producer_fold_strategy: object | None = None,
    query_consumer_stages: object | None = None,
    query_consumer_fusion: object | None = None,
    consumer_stage: object | None = None,
) -> None:
    """Retain a default-only boundary for legacy operator selectors."""
    canonicalize_query_operator_selection(
        query_feature_impl=query_feature_impl,
        query_fold_impl=query_fold_impl,
        query_gradient_flow=query_gradient_flow,
        query_producer_fold_strategy=query_producer_fold_strategy,
        query_consumer_stages=query_consumer_stages,
        query_consumer_fusion=query_consumer_fusion,
    )
    if consumer_stage is not None:
        raise ValueError("consumer_stage: OP-4 exploration is retired")


def legacy_query_consumer_fusion_for_stages(stages: Sequence[str]) -> str:
    """Return the sole retained compatibility spelling."""
    _canonical_consumer_stages(stages)
    return "none"


class _HDParallelBlockPlanContract:
    """Nominal cache-admission view; operator selectors belong to the plan value."""

    cache_before_manifest: tuple[object, ...]
    projected_cache_after_manifest: tuple[object, ...]
    pair_layout_id: str
    device: str
    pair_metadata_materialization_bytes: int


__all__: tuple[str, ...] = ()
