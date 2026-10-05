"""Allocation-free tensor-input planning for causal HD Block-GEMM.

Inspect geometry, resolve execution selectors/backend and compile physical
storage. Immutable plan values live in hd_block_gemm_plan and remain available
through this historical facade, including its explicit private runtime seams.
"""

from __future__ import annotations

from math import ceil
from typing import Sequence

import torch

from .hd_cublas_compat import (
    HdContractionBackendIdentity,
    resolve_backend_for_plan,
)
from .hd_block_gemm_buffers import (
    HDLogicalBuffer,
    _DTYPE_BYTES,
    _EXECUTION_STAGES,
    _FEATURE_STORAGE,
    _STAGE_INDEX,
    _byte_summaries,
    _expected_logical_buffers,
    _peak_bytes,
)
from .hd_block_gemm_cache import (
    CanonicalPairLayout,
    PairMetadataCacheEntry,
    _PAIR_CACHE_ACTIVE_LEASES,
    _PAIR_CACHE_PENDING_MATERIALIZATIONS,
    _PAIR_TENSOR_CACHE,
    _PAIR_TENSOR_CACHE_LOCK,
    _PairCacheAdmissionLease,
    _abandon_pair_cache_admission_lease,
    _acquire_pair_cache_admission_lease,
    _canonical_runtime_device,
    _materialize_pair_layout,
    _pair_cache_manifest_locked,
    _project_cache_manifest,
    _release_pair_cache_admission_lease,
    _validate_pair_cache_admission_lease,
    _validate_pair_cache_admission_lease_locked,
    canonical_pair_layout,
)
from .hd_block_gemm_contracts import (
    _HDParallelBlockPlanContract,
    _MAX_HEAD_DIMENSION,
    _contiguous_strides,
    _require_plain_int,
    _require_head_dimension,
    _sha256,
    _stable_json,
    _validate_backward_schedule,
    _validate_backward_normalize_impl,
    _validate_kv_cross_impl,
    _validate_query_feature_token_tile,
    _validate_query_fold_token_tile,
    _validate_query_fold_input,
    _validate_gradient_staging,
    _validate_forward_normalize_impl,
    _validate_feature_padding,
    _validate_key_feature_impl,
    _validate_key_fold_impl,
    _validate_query_operator_identity,
    _validate_key_retention,
    _validate_save_local_score,
    legacy_query_consumer_fusion_for_stages,
)
from .hd_block_gemm_plan import (
    HDParallelBlockPlan,
    _PHYSICAL_PATH,
    _INPUT_LAYOUT,
    _SUPPORTED_PRECISIONS,
    _INPUT_DTYPE_NAMES,
    _FULL_AUX_RESULT_CONTRACT,
    _NORMALIZED_OUTPUT_ONLY_RESULT_CONTRACT,
    _SUPPORTED_RESULT_CONTRACTS,
    _feature_storage_for_precision,
    _validate_cache_manifest,
)


def _canonical_dtype_name(dtype: torch.dtype) -> str:
    name = str(dtype).removeprefix("torch.")
    if name not in _INPUT_DTYPE_NAMES:
        raise TypeError(
            "Q, K, and V must use float16, bfloat16, float32, or float64 dtype"
        )
    return name


def _canonical_output_dtype_name(
    output_dtype: torch.dtype | None,
    *,
    input_dtype: str,
) -> str:
    if output_dtype is None:
        return input_dtype
    if not isinstance(output_dtype, torch.dtype):
        raise TypeError("output_dtype must be a torch.dtype")
    name = str(output_dtype).removeprefix("torch.")
    if name not in _INPUT_DTYPE_NAMES:
        raise TypeError("output_dtype must be float16, bfloat16, float32, or float64")
    return name


def _read_cuda_tf32_disabled() -> bool:
    """Read installed CUDA matmul policy APIs and fail closed if unavailable."""
    matmul = torch.backends.cuda.matmul
    observed_policy = False
    disabled = True
    try:
        fp32_precision = getattr(matmul, "fp32_precision")
    except (AttributeError, RuntimeError):
        pass
    else:
        observed_policy = True
        disabled = disabled and fp32_precision == "ieee"
    try:
        allow_tf32 = getattr(matmul, "allow_tf32")
    except (AttributeError, RuntimeError):
        pass
    else:
        observed_policy = True
        disabled = disabled and allow_tf32 is False
    return observed_policy and disabled


def _require_precision_policy(
    device_type: str,
    precision: str,
    *,
    input_dtype: str | None = None,
) -> None:
    if precision not in _SUPPORTED_PRECISIONS:
        raise ValueError("unsupported HD Block-GEMM precision")
    if precision == "bf16_tensorcore":
        if device_type != "cuda":
            raise ValueError("bf16_tensorcore requires CUDA")
        if input_dtype != "bfloat16":
            raise ValueError("bf16_tensorcore requires bfloat16 Q/K/V")
        return
    if device_type != "cuda":
        return
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError(
            "CUDA fp32_ieee requires ambient float32 matmul precision 'highest'"
        )
    if not _read_cuda_tf32_disabled():
        raise RuntimeError("CUDA fp32_ieee requires TF32 matmul to be disabled")


def _validate_geometry(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    str,
    str,
    str,
    tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
]:
    if not all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
        raise TypeError("Q, K, and V must be tensors")
    if any(tensor.ndim != 4 for tensor in (q, k, v)):
        raise ValueError("Q, K, and V must be rank-4 tensors")
    if any(dimension <= 0 for tensor in (q, k, v) for dimension in tensor.shape):
        raise ValueError("Q, K, and V geometry sizes must be positive")
    if any(tensor.layout != torch.strided for tensor in (q, k, v)):
        raise ValueError("Q, K, and V must use torch.strided layout")
    input_strides = tuple(tuple(tensor.stride()) for tensor in (q, k, v))
    expected_strides = tuple(
        _contiguous_strides(tuple(tensor.shape)) for tensor in (q, k, v)
    )
    if input_strides != expected_strides:
        raise ValueError("Q, K, and V must use exact row-major contiguous layout")
    batch_size, query_heads, sequence_length, head_dimension = q.shape
    k_batch, key_value_heads, k_sequence, k_head_dimension = k.shape
    v_batch, v_heads, v_sequence, value_dimension = v.shape
    if batch_size != k_batch or batch_size != v_batch:
        raise ValueError("Q, K, and V must have the same batch size")
    if sequence_length != k_sequence or sequence_length != v_sequence:
        raise ValueError("Q, K, and V must have the same sequence length")
    if key_value_heads != v_heads:
        raise ValueError("K and V must have the same key/value head count")
    if query_heads % key_value_heads:
        raise ValueError("query heads must be divisible by key/value heads")
    if head_dimension != k_head_dimension:
        raise ValueError("Q and K must have the same head dimension")
    _require_head_dimension(head_dimension)
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q, K, and V must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("Q, K, and V must have the same dtype")
    runtime_device = _canonical_runtime_device(
        q.device,
        name="Q, K, and V tensor device",
    )
    dtype_name = _canonical_dtype_name(q.dtype)
    return (
        batch_size,
        query_heads,
        key_value_heads,
        sequence_length,
        head_dimension,
        value_dimension,
        dtype_name,
        runtime_device.type,
        str(runtime_device),
        input_strides,
    )


def build_hd_block_gemm_plan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    token_block: int = 128,
    feature_wave_blocks: int | None = None,
    precision: str = "fp32_ieee",
    result_contract: str = _FULL_AUX_RESULT_CONTRACT,
    key_feature_impl: str = "generic_materialized",
    key_fold_impl: str = "generic_materialized",
    query_feature_impl: str = "generic_materialized",
    query_feature_token_tile: int = 1,
    query_fold_impl: str = "generic_materialized",
    query_fold_token_tile: int = 1,
    query_fold_input: str = "staged_fp32",
    query_gradient_flow: str = "materialized",
    query_producer_fold_strategy: str | None = None,
    query_producer_partition_count: int | None = None,
    query_consumer_stages: Sequence[str] | None = None,
    query_consumer_fusion: str | None = None,
    backward_schedule: str = "split",
    gradient_staging: str = "per_wave",
    forward_normalize_impl: str = "torch",
    backward_normalize_impl: str = "torch",
    kv_cross_impl: str = "split",
    save_local_score: bool = False,
    key_retention: str = "none",
    output_dtype: torch.dtype | None = None,
    requested_gradient_mask: tuple[bool, bool, bool, bool, bool, bool],
    memory_budget_bytes: int | None = None,
    model_residency_bytes: int = 0,
    strict_backend: bool = True,
    feature_padding: str = "none",
) -> HDParallelBlockPlan:
    """Build an allocation-free, exact logical plan for private Block-GEMM."""
    if not isinstance(result_contract, str):
        raise TypeError("result_contract must be a string")
    if result_contract not in _SUPPORTED_RESULT_CONTRACTS:
        raise ValueError("unsupported HD Block-GEMM result contract")
    if result_contract != _FULL_AUX_RESULT_CONTRACT:
        raise NotImplementedError(
            "normalized_output_only is an unimplemented private specialization"
        )
    (
        batch_size,
        query_heads,
        key_value_heads,
        sequence_length,
        head_dimension,
        value_dimension,
        input_dtype,
        device_type,
        device,
        input_strides,
    ) = _validate_geometry(q, k, v)
    feature_padding = _validate_feature_padding(
        feature_padding,
        head_dimension=head_dimension,
    )
    resolved_output_dtype = _canonical_output_dtype_name(
        output_dtype,
        input_dtype=input_dtype,
    )
    token_block = _require_plain_int(token_block, name="token_block", minimum=1)
    number_blocks = ceil(sequence_length / token_block)
    if feature_wave_blocks is None:
        feature_wave_blocks = number_blocks
    feature_wave_blocks = _require_plain_int(
        feature_wave_blocks,
        name="feature_wave_blocks",
        minimum=1,
    )
    if feature_wave_blocks > number_blocks:
        raise ValueError("feature_wave_blocks cannot exceed the number of blocks")
    if not isinstance(precision, str):
        raise TypeError("precision must be a string")
    if type(strict_backend) is not bool:
        raise TypeError("strict_backend must be a bool")
    if not isinstance(requested_gradient_mask, tuple):
        raise TypeError("requested_gradient_mask must be a tuple")
    if len(requested_gradient_mask) != 6:
        raise ValueError("requested_gradient_mask must contain exactly six bools")
    if any(type(requested) is not bool for requested in requested_gradient_mask):
        raise TypeError("requested_gradient_mask entries must be bool values")
    if memory_budget_bytes is not None:
        memory_budget_bytes = _require_plain_int(
            memory_budget_bytes,
            name="memory_budget_bytes",
            minimum=1,
        )
    model_residency_bytes = _require_plain_int(
        model_residency_bytes,
        name="model_residency_bytes",
        minimum=0,
    )
    resolved_backend = resolve_backend_for_plan(
        precision,
        torch.device(device),
        strict_backend=strict_backend,
    )
    requested_precision = resolved_backend.requested_precision
    precision = resolved_backend.effective_precision
    _require_precision_policy(
        device_type,
        precision,
        input_dtype=input_dtype,
    )
    canonical_query_config = _validate_query_operator_identity(
        query_feature_impl=query_feature_impl,
        query_fold_impl=query_fold_impl,
        query_gradient_flow=query_gradient_flow,
        query_producer_fold_strategy=query_producer_fold_strategy,
        query_consumer_stages=query_consumer_stages,
        query_consumer_fusion=query_consumer_fusion,
        precision=precision,
        head_dimension=head_dimension,
        result_contract=result_contract,
    )
    query_feature_impl = str(canonical_query_config["query_feature_impl"])
    query_feature_token_tile = _validate_query_feature_token_tile(
        query_feature_token_tile,
        query_feature_impl=query_feature_impl,
    )
    query_fold_impl = str(canonical_query_config["query_fold_impl"])
    query_gradient_flow = str(canonical_query_config["query_gradient_flow"])
    query_fold_token_tile = _validate_query_fold_token_tile(
        query_fold_token_tile,
        query_fold_impl=query_fold_impl,
    )
    query_fold_input = _validate_query_fold_input(
        query_fold_input,
        query_fold_impl=query_fold_impl,
        query_gradient_flow=query_gradient_flow,
    )
    query_producer_fold_strategy = str(
        canonical_query_config["query_producer_fold_strategy"]
    )
    query_consumer_stages = tuple(canonical_query_config["query_consumer_stages"])
    key_feature_impl = _validate_key_feature_impl(
        key_feature_impl=key_feature_impl,
        precision=precision,
        head_dimension=head_dimension,
        result_contract=result_contract,
    )
    key_fold_impl = _validate_key_fold_impl(
        key_fold_impl=key_fold_impl,
        precision=precision,
        head_dimension=head_dimension,
        result_contract=result_contract,
    )
    if query_producer_partition_count is not None:
        raise ValueError("query_producer_partition_count: OP-3 exploration is retired")
    backward_schedule = _validate_backward_schedule(
        backward_schedule,
        requested_gradient_mask=requested_gradient_mask,
        query_gradient_flow=query_gradient_flow,
        query_consumer_stages=query_consumer_stages,
    )
    gradient_staging = _validate_gradient_staging(
        gradient_staging,
        precision=precision,
    )
    forward_normalize_impl = _validate_forward_normalize_impl(
        forward_normalize_impl,
        precision=precision,
    )
    backward_normalize_impl = _validate_backward_normalize_impl(
        backward_normalize_impl,
        precision=precision,
    )
    kv_cross_impl = _validate_kv_cross_impl(
        kv_cross_impl,
        precision=precision,
    )
    save_local_score = _validate_save_local_score(
        save_local_score,
        requested_gradient_mask=requested_gradient_mask,
        precision=precision,
    )
    requested_key_retention = _validate_key_retention(
        key_retention,
        precision=precision,
    )
    key_retention = _validate_key_retention(
        requested_key_retention,
        precision=precision,
        requested_gradient_mask=requested_gradient_mask,
    )

    layout = canonical_pair_layout(head_dimension)
    pair_count = layout.pair_count
    current_cache_entry = PairMetadataCacheEntry(
        layout_id=layout.layout_id,
        device=device,
        pair_count=pair_count,
    )
    with _PAIR_TENSOR_CACHE_LOCK:
        cache_before_manifest = _pair_cache_manifest_locked()
    projected_cache_after_manifest = _project_cache_manifest(
        cache_before_manifest,
        current_cache_entry,
    )
    feature_dimension = 1 + head_dimension + pair_count
    physical_feature_dimension = (
        2160 if feature_padding == "f2160" else feature_dimension
    )
    augmented_value_dimension = value_dimension + 1
    logical_buffers = _expected_logical_buffers(
        geometry=(
            batch_size,
            query_heads,
            key_value_heads,
            sequence_length,
            head_dimension,
            value_dimension,
        ),
        token_block=token_block,
        feature_wave_blocks=feature_wave_blocks,
        input_dtype=input_dtype,
        output_dtype=resolved_output_dtype,
        pair_layout_id=layout.layout_id,
        device=device,
        projected_cache_after_manifest=projected_cache_after_manifest,
        requested_gradient_mask=requested_gradient_mask,
        precision=precision,
        result_contract=result_contract,
        key_feature_impl=key_feature_impl,
        key_fold_impl=key_fold_impl,
        query_feature_impl=query_feature_impl,
        query_fold_impl=query_fold_impl,
        query_fold_input=query_fold_input,
        query_gradient_flow=query_gradient_flow,
        query_producer_fold_strategy=query_producer_fold_strategy,
        query_producer_partition_count=query_producer_partition_count,
        query_consumer_stages=query_consumer_stages,
        backward_schedule=backward_schedule,
        gradient_staging=gradient_staging,
        backward_normalize_impl=backward_normalize_impl,
        save_local_score=save_local_score,
        key_retention=key_retention,
        physical_feature_dimension=physical_feature_dimension,
    )
    summaries = _byte_summaries(
        logical_buffers,
        requested_gradient_mask,
        cache_before_manifest,
    )
    projected_live_bytes = summaries["invocation_peak_bytes"] + model_residency_bytes
    if memory_budget_bytes is not None and projected_live_bytes > memory_budget_bytes:
        raise ValueError(
            "memory budget is below the exact invocation peak plus model residency"
        )

    return HDParallelBlockPlan(
        physical_path=_PHYSICAL_PATH,
        token_block=token_block,
        feature_wave_blocks=feature_wave_blocks,
        feature_storage=_feature_storage_for_precision(precision),
        result_contract=result_contract,
        key_feature_impl=key_feature_impl,
        key_fold_impl=key_fold_impl,
        query_feature_impl=query_feature_impl,
        query_feature_token_tile=query_feature_token_tile,
        query_fold_impl=query_fold_impl,
        query_fold_token_tile=query_fold_token_tile,
        query_fold_input=query_fold_input,
        query_gradient_flow=query_gradient_flow,
        query_producer_fold_strategy=query_producer_fold_strategy,
        query_producer_partition_count=query_producer_partition_count,
        query_consumer_stages=query_consumer_stages,
        backward_schedule=backward_schedule,
        gradient_staging=gradient_staging,
        forward_normalize_impl=forward_normalize_impl,
        backward_normalize_impl=backward_normalize_impl,
        kv_cross_impl=kv_cross_impl,
        save_local_score=save_local_score,
        requested_key_retention=requested_key_retention,
        key_retention=key_retention,
        requested_precision=requested_precision,
        precision=precision,
        precision_fallback_reason=resolved_backend.fallback_reason,
        contraction_backend_identity=resolved_backend.identity,
        requested_gradient_mask=requested_gradient_mask,
        geometry=(
            batch_size,
            query_heads,
            key_value_heads,
            sequence_length,
            head_dimension,
            value_dimension,
        ),
        gqa_ratio=query_heads // key_value_heads,
        pair_layout_id=layout.layout_id,
        pair_count=pair_count,
        feature_dimension=feature_dimension,
        feature_padding=feature_padding,
        physical_feature_dimension=physical_feature_dimension,
        augmented_value_dimension=augmented_value_dimension,
        physical_augmented_value_dimension=augmented_value_dimension,
        number_blocks=number_blocks,
        input_dtype=input_dtype,
        input_layout=_INPUT_LAYOUT,
        input_strides=input_strides,
        device_type=device_type,
        device=device,
        cache_before_manifest=cache_before_manifest,
        projected_cache_after_manifest=projected_cache_after_manifest,
        memory_budget_bytes=memory_budget_bytes,
        model_residency_bytes=model_residency_bytes,
        execution_stages=_EXECUTION_STAGES,
        logical_buffers=logical_buffers,
        **summaries,
        projected_live_bytes=projected_live_bytes,
    )


__all__ = (
    "CanonicalPairLayout",
    "HDLogicalBuffer",
    "HDParallelBlockPlan",
    "PairMetadataCacheEntry",
    "build_hd_block_gemm_plan",
    "canonical_pair_layout",
)


# Narrow lazy seam for explicit candidate tests/adapters.  Keeping the import
# inside the wrapper lets the runtime itself be imported first without a
# partially initialized planner/runtime cycle.  Public routing and the
# module's public export surface intentionally remain unchanged.
def _hd_parallel_block_gemm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    scale: float,
    eps: float,
    layout: CanonicalPairLayout,
    plan: HDParallelBlockPlan,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from .hd_block_gemm_runtime import hd_parallel_block_gemm

    return hd_parallel_block_gemm(
        q,
        k,
        v,
        a,
        b,
        c,
        scale=scale,
        eps=eps,
        layout=layout,
        plan=plan,
    )


def _hd_parallel_block_gemm_autograd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    scale: float,
    eps: float,
    layout: CanonicalPairLayout,
    plan: HDParallelBlockPlan,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the private first-order autograd candidate without public routing."""
    from .hd_block_gemm_autograd import _hd_parallel_block_gemm_autograd as run

    return run(
        q,
        k,
        v,
        a,
        b,
        c,
        scale=scale,
        eps=eps,
        layout=layout,
        plan=plan,
    )
