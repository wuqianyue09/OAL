"""OAL grouped coefficients for the causal HD attention API.

Translate grouped PSD coefficients into logical A/B/C and run the planned
Block-GEMM core for inference or first-order training.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import torch

from .hd_block_gemm import (
    HDParallelBlockPlan,
    _hd_parallel_block_gemm,
    _hd_parallel_block_gemm_autograd,
    build_hd_block_gemm_plan,
    canonical_pair_layout,
)
from .hd_block_gemm_profiling import _record_hd_stage

_POSITIVE_FP32_FLOOR = torch.finfo(torch.float32).tiny * torch.finfo(torch.float32).eps


def _requested_gradient_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    coefficient_source_requires_grad: bool,
) -> tuple[bool, bool, bool, bool, bool, bool]:
    live = torch.is_grad_enabled()
    coefficient_grad = live and coefficient_source_requires_grad
    return (
        live and q.requires_grad,
        live and k.requires_grad,
        live and v.requires_grad,
        coefficient_grad,
        coefficient_grad,
        coefficient_grad,
    )


def _resolve_scale(scale: float | None, head_dimension: int) -> float:
    if scale is None:
        return head_dimension**-0.5
    if isinstance(scale, bool) or not isinstance(scale, Real):
        raise TypeError("scale must be a finite real scalar or None")
    resolved = float(scale)
    if not math.isfinite(resolved):
        raise ValueError("scale must be finite")
    return resolved


def _expand_factor_coefficients_tensorcore(
    factor: torch.Tensor,
    *,
    query_heads: int,
    gmax: int,
    kernel_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Form the existing MGQ PSD coefficients with FP32 pointwise reduction."""
    factor_by_head = factor.unsqueeze(0) if factor.ndim == 1 else factor
    factor_by_head = factor_by_head.to(dtype=torch.float32)
    if factor_by_head.shape[0] == 1:
        factor_by_head = factor_by_head.expand(query_heads, -1)
    side = gmax + 1
    rows, columns = torch.tril_indices(side, side, device=factor.device)
    linear_indices = rows * side + columns
    lower_flat = torch.zeros(
        (query_heads, side * side),
        dtype=torch.float32,
        device=factor.device,
    ).scatter(
        1,
        linear_indices.unsqueeze(0).expand(query_heads, -1),
        factor_by_head,
    )
    lower = lower_flat.reshape(query_heads, side, side)
    matrix = torch.sum(
        lower.unsqueeze(2) * lower.unsqueeze(1),
        dim=-1,
        dtype=torch.float32,
    )
    epsilon = torch.as_tensor(
        kernel_eps,
        dtype=torch.float32,
        device=factor.device,
    )
    constant = matrix[:, 0, 0] + epsilon
    linear = 2.0 * matrix[:, 0, 1:]
    quadratic = matrix[:, 1:, 1:].contiguous()
    return constant, linear, quadratic


@dataclass(frozen=True)
class _PreparedGroupedHdBlockGemm:
    """Immutable structural state for repeated private MGQ invocations.

    The snapshot deliberately excludes factor-derived coefficients and all
    invocation storage.  Factors remain live autograd inputs, while every run
    still enters the existing plan-bound runtime and acquires its own cache
    lease/backend token.
    """

    plan: HDParallelBlockPlan
    groups: torch.Tensor
    group_indices: torch.Tensor
    pair_row_groups: torch.Tensor
    pair_column_groups: torch.Tensor
    head_indices: torch.Tensor
    gmax: int
    scale: float
    kernel_eps: float
    normalization_floor: float


def prepare_grouped_hd_block_gemm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    factor: torch.Tensor,
    *,
    scale: float | None = None,
    kernel_eps: float = 1e-6,
    token_block: int = 128,
    feature_wave_blocks: int | None = None,
    memory_budget_bytes: int | None = None,
    model_residency_bytes: int = 0,
    precision: str = "fp32_ieee",
    key_feature_impl: str = "generic_materialized",
    key_fold_impl: str = "generic_materialized",
    query_feature_impl: str = "generic_materialized",
    query_feature_token_tile: int = 1,
    query_fold_impl: str = "generic_materialized",
    query_fold_token_tile: int = 1,
    query_fold_input: str = "staged_fp32",
    query_gradient_flow: str = "materialized",
    query_producer_fold_strategy: str | None = None,
    query_consumer_stages: tuple[str, ...] | None = None,
    query_consumer_fusion: str | None = None,
    backward_schedule: str = "split",
    gradient_staging: str = "per_wave",
    forward_normalize_impl: str = "torch",
    backward_normalize_impl: str = "torch",
    kv_cross_impl: str = "split",
    save_local_score: bool = False,
    key_retention: str = "none",
    feature_padding: str = "none",
) -> _PreparedGroupedHdBlockGemm:
    """Freeze MGQ geometry/groups once for an explicitly repeated workload."""
    from .grouped_quadratic import (
        _canonicalize_dim_groups,
        _resolve_working_dtype,
        _validate_factor,
        _validate_kernel_eps,
        _validate_qkv,
    )

    q, k, v = _validate_qkv(q, k, v)
    factor, gmax = _validate_factor(factor, query_heads=q.shape[1], q_device=q.device)
    working_dtype = _resolve_working_dtype(q.dtype, factor.dtype)
    if working_dtype != torch.float32:
        raise TypeError(
            "private grouped candidate currently requires FP32 working precision"
        )
    resolved_eps = _validate_kernel_eps(kernel_eps, working_dtype=working_dtype)
    canonical_groups = _canonicalize_dim_groups(
        dim_groups,
        query_heads=q.shape[1],
        head_dimension=q.shape[-1],
        gmax=gmax,
        q_device=q.device,
    )
    gradient_mask = _requested_gradient_mask(
        q,
        k,
        v,
        coefficient_source_requires_grad=factor.requires_grad,
    )
    plan = build_hd_block_gemm_plan(
        q,
        k,
        v,
        token_block=token_block,
        feature_wave_blocks=feature_wave_blocks,
        memory_budget_bytes=memory_budget_bytes,
        model_residency_bytes=model_residency_bytes,
        requested_gradient_mask=gradient_mask,
        precision=precision,
        key_feature_impl=key_feature_impl,
        key_fold_impl=key_fold_impl,
        query_feature_impl=query_feature_impl,
        query_feature_token_tile=query_feature_token_tile,
        query_fold_impl=query_fold_impl,
        query_fold_token_tile=query_fold_token_tile,
        query_fold_input=query_fold_input,
        query_gradient_flow=query_gradient_flow,
        query_producer_fold_strategy=query_producer_fold_strategy,
        query_consumer_stages=query_consumer_stages,
        query_consumer_fusion=query_consumer_fusion,
        backward_schedule=backward_schedule,
        gradient_staging=gradient_staging,
        forward_normalize_impl=forward_normalize_impl,
        backward_normalize_impl=backward_normalize_impl,
        kv_cross_impl=kv_cross_impl,
        save_local_score=save_local_score,
        key_retention=key_retention,
        feature_padding=feature_padding,
    )
    groups = (
        (
            canonical_groups.unsqueeze(0).expand(q.shape[1], -1)
            if canonical_groups.ndim == 1
            else canonical_groups
        )
        .detach()
        .clone()
    )
    group_indices = groups.to(dtype=torch.int64)
    layout = canonical_pair_layout(plan.head_dimension)
    rows = torch.tensor(layout.rows, dtype=torch.int64, device=q.device)
    columns = torch.tensor(layout.columns, dtype=torch.int64, device=q.device)
    return _PreparedGroupedHdBlockGemm(
        plan=plan,
        groups=groups,
        group_indices=group_indices,
        pair_row_groups=group_indices.index_select(1, rows),
        pair_column_groups=group_indices.index_select(1, columns),
        head_indices=torch.arange(q.shape[1], device=q.device)[:, None],
        gmax=gmax,
        scale=_resolve_scale(scale, q.shape[-1]),
        kernel_eps=resolved_eps,
        normalization_floor=_POSITIVE_FP32_FLOOR,
    )


def run_prepared_grouped_hd_block_gemm(
    prepared: _PreparedGroupedHdBlockGemm,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    factor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one live MGQ invocation against explicit immutable preparation."""
    from .grouped_quadratic import (
        _expand_factor_coefficients,
        _resolve_working_dtype,
        _validate_factor,
        _validate_qkv,
    )

    if not isinstance(prepared, _PreparedGroupedHdBlockGemm):
        raise TypeError(
            "prepared must be an explicit grouped HD Block-GEMM preparation"
        )
    q, k, v = _validate_qkv(q, k, v)
    factor, gmax = _validate_factor(factor, query_heads=q.shape[1], q_device=q.device)
    if gmax != prepared.gmax:
        raise ValueError("factor group geometry does not match the preparation")
    working_dtype = _resolve_working_dtype(q.dtype, factor.dtype)
    if working_dtype != torch.float32:
        raise TypeError(
            "private grouped candidate currently requires FP32 working precision"
        )
    gradient_mask = _requested_gradient_mask(
        q,
        k,
        v,
        coefficient_source_requires_grad=factor.requires_grad,
    )
    if gradient_mask != prepared.plan.requested_gradient_mask:
        raise ValueError("live gradient mode does not match the preparation")

    with _record_hd_stage("hd.forward.coefficients"):
        if prepared.plan.precision == "bf16_tensorcore":
            a, linear_by_group, quadratic_by_groups = (
                _expand_factor_coefficients_tensorcore(
                    factor,
                    query_heads=prepared.plan.query_heads,
                    gmax=prepared.gmax,
                    kernel_eps=prepared.kernel_eps,
                )
            )
        else:
            a, linear_by_group, quadratic_by_groups = _expand_factor_coefficients(
                factor,
                query_heads=prepared.plan.query_heads,
                gmax=prepared.gmax,
                kernel_eps=prepared.kernel_eps,
                working_dtype=torch.float32,
            )
        b = linear_by_group.gather(1, prepared.group_indices)
        c = quadratic_by_groups[
            prepared.head_indices,
            prepared.pair_row_groups,
            prepared.pair_column_groups,
        ]
    return _run_logical_coefficients(
        q,
        k,
        v,
        a.contiguous(),
        b.contiguous(),
        c.contiguous(),
        plan=prepared.plan,
        scale=prepared.scale,
        normalization_floor=prepared.normalization_floor,
    )


def _run_logical_coefficients(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    plan: HDParallelBlockPlan,
    scale: float,
    normalization_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    layout = canonical_pair_layout(plan.head_dimension)
    run = (
        _hd_parallel_block_gemm_autograd
        if any(plan.requested_gradient_mask)
        else _hd_parallel_block_gemm
    )
    return run(
        q,
        k,
        v,
        a,
        b,
        c,
        scale=scale,
        eps=normalization_floor,
        layout=layout,
        plan=plan,
    )


def grouped_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    factor: torch.Tensor,
    *,
    scale: float | None = None,
    kernel_eps: float = 1e-6,
    token_block: int = 128,
    feature_wave_blocks: int | None = None,
    memory_budget_bytes: int | None = None,
    model_residency_bytes: int = 0,
    precision: str = "fp32_ieee",
    key_feature_impl: str = "generic_materialized",
    key_fold_impl: str = "generic_materialized",
    query_feature_impl: str = "generic_materialized",
    query_feature_token_tile: int = 1,
    query_fold_impl: str = "generic_materialized",
    query_fold_token_tile: int = 1,
    query_fold_input: str = "staged_fp32",
    query_gradient_flow: str = "materialized",
    query_producer_fold_strategy: str | None = None,
    query_consumer_stages: tuple[str, ...] | None = None,
    query_consumer_fusion: str | None = None,
    backward_schedule: str = "split",
    gradient_staging: str = "per_wave",
    forward_normalize_impl: str = "torch",
    backward_normalize_impl: str = "torch",
    kv_cross_impl: str = "split",
    save_local_score: bool = False,
    key_retention: str = "none",
    feature_padding: str = "none",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare and run one grouped causal invocation with a live factor."""
    prepared = prepare_grouped_hd_block_gemm(
        q,
        k,
        v,
        dim_groups,
        factor,
        scale=scale,
        kernel_eps=kernel_eps,
        token_block=token_block,
        feature_wave_blocks=feature_wave_blocks,
        memory_budget_bytes=memory_budget_bytes,
        model_residency_bytes=model_residency_bytes,
        precision=precision,
        key_feature_impl=key_feature_impl,
        key_fold_impl=key_fold_impl,
        query_feature_impl=query_feature_impl,
        query_feature_token_tile=query_feature_token_tile,
        query_fold_impl=query_fold_impl,
        query_fold_token_tile=query_fold_token_tile,
        query_fold_input=query_fold_input,
        query_gradient_flow=query_gradient_flow,
        query_producer_fold_strategy=query_producer_fold_strategy,
        query_consumer_stages=query_consumer_stages,
        query_consumer_fusion=query_consumer_fusion,
        backward_schedule=backward_schedule,
        gradient_staging=gradient_staging,
        forward_normalize_impl=forward_normalize_impl,
        backward_normalize_impl=backward_normalize_impl,
        kv_cross_impl=kv_cross_impl,
        save_local_score=save_local_score,
        key_retention=key_retention,
        feature_padding=feature_padding,
    )
    return run_prepared_grouped_hd_block_gemm(prepared, q, k, v, factor)


# Repeated training/inference owners explicitly retain this opaque preparation
# between compatible calls. The implementation type remains private: callers
# can execute it, but do not receive mutable group/index metadata as API.
# Existing private callers retain their historical names. New repeated-call
# owners import the named API above.
_grouped_hd_block_gemm_forward = grouped_causal_attention
_prepare_grouped_hd_block_gemm = prepare_grouped_hd_block_gemm
_run_prepared_grouped_hd_block_gemm = run_prepared_grouped_hd_block_gemm


__all__ = (
    "grouped_causal_attention",
    "prepare_grouped_hd_block_gemm",
    "run_prepared_grouped_hd_block_gemm",
)
