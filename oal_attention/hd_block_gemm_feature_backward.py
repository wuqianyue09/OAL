"""Vectorized packed-feature gradient folds and coefficient reductions."""

from __future__ import annotations

from dataclasses import replace
import math

import torch

from .hd_block_gemm_feature_context import (
    _BACKWARD,
    _COEFFICIENT_ACCUMULATING,
    _COEFFICIENT_FINALIZED,
    _COEFFICIENT_IDLE,
    _PreparedFeatureContext,
    _copy_into,
    _copy_wave_to_work,
    _ensure_backward_workspace,
    _ensure_pairs,
    _ensure_prepared_values,
    _head_feature_view,
    _new_coefficient_coverage,
    _pair_metadata_view,
    _require_coefficient_block_wave,
    _require_feature_gradient,
    _require_feature_wave,
    _require_no_grad_tensors,
    _require_prepared,
    _require_raw_values,
    _require_scatter_mode,
    _scatter_pair_source,
    _wave_view,
)


def _require_planned_query_input_work(
    q: object,
    *,
    plan: object,
) -> tuple[torch.Tensor, int]:
    if not isinstance(q, torch.Tensor):
        raise TypeError("planned Q work must be a tensor")
    if q.layout != torch.strided or q.ndim < 3:
        raise ValueError("planned Q work must have shape [B,Hq,...,D]")
    if q.shape[0] != plan.batch_size or q.shape[1] != plan.query_heads:
        raise ValueError("planned Q work leading dimensions do not match the plan")
    if q.shape[-1] != plan.head_dimension:
        raise ValueError("planned Q work feature dimension does not match the plan")
    if q.dtype != torch.float32:
        raise ValueError("planned Q work must use FP32 storage")
    if str(q.device) != plan.device:
        raise ValueError("planned Q work device does not match the plan")
    token_count = math.prod(q.shape[2:-1])
    if token_count > plan.feature_wave_blocks * plan.token_block:
        raise ValueError("planned Q work exceeds the feature wave capacity")
    return q, token_count


def _fold_key_feature_gradient(
    k: torch.Tensor | None,
    d_phi_k: torch.Tensor | None,
    *,
    prepared: _PreparedFeatureContext,
) -> torch.Tensor | None:
    """Fold a complete key-feature gradient with the selected implementation."""
    prepared = _require_prepared(prepared, mode=_BACKWARD)
    plan = prepared.plan
    if not plan.requested_gradient_mask[1]:
        return None
    raw_values = _require_raw_values(prepared)
    _require_no_grad_tensors(k, d_phi_k, raw_values.a, raw_values.b, raw_values.c)
    k, token_count = _require_feature_wave(
        k, name="K", head_count=plan.key_value_heads, plan=plan
    )
    d_phi_k, gradient_token_count = _require_feature_gradient(
        d_phi_k, name="dPhiK", head_count=plan.key_value_heads, plan=plan
    )
    if gradient_token_count != token_count or d_phi_k.shape[:-1] != k.shape[:-1]:
        raise ValueError("dPhiK leading dimensions must match K")
    if plan.key_fold_impl == "generic_materialized":
        _require_scatter_mode(prepared.device)
    _ensure_prepared_values(prepared)
    _ensure_backward_workspace(prepared, key_fold=True)
    workspace = prepared.backward_workspace
    if workspace is None or workspace.key_output is None:
        raise RuntimeError("backward key workspace was not initialized")
    try:
        output = _wave_view(
            workspace.key_output,
            wave_shape=k.shape[:-1],
            token_count=token_count,
            feature_dimension=plan.head_dimension,
        )
        if plan.key_fold_impl == "triton_materialized":
            from .hd_block_gemm_key_ops import fold_triton_key_feature_gradient

            return fold_triton_key_feature_gradient(
                k,
                d_phi_k,
                output,
                plan=plan,
            )
        _ensure_pairs(prepared)
        pairs = prepared.pairs
        if (
            workspace.key_input_work is None
            or workspace.key_pair_source is None
            or pairs is None
        ):
            raise RuntimeError("generic backward key workspace was not initialized")
        k_work = _copy_wave_to_work(
            workspace.key_input_work, k, token_count=token_count
        )
        source = _wave_view(
            workspace.key_pair_source,
            wave_shape=k.shape[:-1],
            token_count=token_count,
            feature_dimension=plan.pair_count,
        )
        pair_start = 1 + plan.head_dimension
        _copy_into(output, d_phi_k[..., 1:pair_start])
        torch.index_select(k_work, -1, pairs.columns, out=source)
        source.mul_(d_phi_k[..., pair_start : pair_start + plan.pair_count])
        _scatter_pair_source(output, pairs.rows, source)
        torch.index_select(k_work, -1, pairs.rows, out=source)
        source.mul_(d_phi_k[..., pair_start : pair_start + plan.pair_count])
        _scatter_pair_source(output, pairs.columns, source)
        return output
    except Exception:
        prepared._fail()
        raise


def _begin_query_coefficient_gradient_accumulation(
    prepared: _PreparedFeatureContext,
) -> None:
    """Begin one complete, non-overlapping block-partial lifecycle."""
    prepared = _require_prepared(prepared, mode=_BACKWARD)
    need_a, need_b, need_c = prepared.plan.requested_gradient_mask[3:]
    if not (need_a or need_b or need_c):
        raise RuntimeError("no coefficient gradients were requested")
    raw_values = _require_raw_values(prepared)
    _require_no_grad_tensors(raw_values.a, raw_values.b, raw_values.c)
    if prepared.coefficient_state != _COEFFICIENT_IDLE:
        raise RuntimeError("coefficient gradient accumulation already began")
    _ensure_prepared_values(prepared)
    _ensure_backward_workspace(prepared, coefficients=True)
    try:
        local_coverage = _new_coefficient_coverage(prepared.plan.number_blocks)
    except Exception:
        prepared._fail()
        raise
    prepared.coefficient_coverage = local_coverage
    prepared.coefficient_state = _COEFFICIENT_ACCUMULATING


def _fold_query_feature_gradient(
    q: torch.Tensor | None,
    d_phi_q: torch.Tensor | None,
    *,
    prepared: _PreparedFeatureContext,
    block_start: int | None = None,
    q_is_planned_work: bool = False,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Fold one query wave; coefficient results are returned by finalize."""
    prepared = _require_prepared(prepared, mode=_BACKWARD)
    plan = prepared.plan
    need_q, _, _, need_a, need_b, need_c = plan.requested_gradient_mask
    if not (need_q or need_a or need_b or need_c):
        return None, None, None, None
    raw_values = _require_raw_values(prepared)
    _require_no_grad_tensors(q, d_phi_q, raw_values.a, raw_values.b, raw_values.c)
    d_phi_q, token_count = _require_feature_gradient(
        d_phi_q, name="dPhiQ", head_count=plan.query_heads, plan=plan
    )
    needs_input_q = need_q or need_b or need_c
    raw_query_input = getattr(plan, "query_fold_input", "staged_fp32") == "raw"
    # An A-only fold has no Q input; dPhiQ still defines its wave capacity.
    q_token_count = token_count
    if needs_input_q:
        if q_is_planned_work:
            q, q_token_count = _require_planned_query_input_work(q, plan=plan)
        else:
            q, q_token_count = _require_feature_wave(
                q, name="Q", head_count=plan.query_heads, plan=plan
            )
        if raw_query_input:
            if q_is_planned_work or q_token_count > token_count:
                raise ValueError("raw query fold input does not match dPhiQ capacity")
        elif q_token_count != token_count or q.shape[:-1] != d_phi_q.shape[:-1]:
            raise ValueError("dPhiQ leading dimensions must match Q")
    coefficient_requested = need_a or need_b or need_c
    if coefficient_requested:
        start, wave_blocks = _require_coefficient_block_wave(
            d_phi_q,
            q,
            block_start=block_start,
            needs_input_q=needs_input_q and not raw_query_input,
            plan=plan,
        )
        if prepared.coefficient_state != _COEFFICIENT_ACCUMULATING:
            raise RuntimeError("coefficient gradient accumulation has not begun")
        coverage = prepared.coefficient_coverage
        if coverage is None:
            raise RuntimeError("coefficient gradient coverage is not initialized")
        if any(coverage[start : start + wave_blocks]):
            raise ValueError("coefficient gradient block range overlaps prior work")
    elif block_start is not None:
        raise ValueError("block_start is only valid for coefficient gradients")
    if need_q and plan.query_fold_impl == "generic_materialized":
        _require_scatter_mode(prepared.device)
    _ensure_prepared_values(prepared)
    _ensure_backward_workspace(prepared, query_fold=True)
    if need_c or (need_q and plan.query_fold_impl == "generic_materialized"):
        _ensure_pairs(prepared)
    workspace = prepared.backward_workspace
    values = prepared.values
    pairs = prepared.pairs
    if workspace is None:
        raise RuntimeError("backward query workspace was not initialized")
    if needs_input_q and not raw_query_input and workspace.query_input_work is None:
        raise RuntimeError("backward query input workspace was not initialized")
    try:
        q_work = None
        if needs_input_q and q is not None:
            if raw_query_input:
                q_work = q
            elif workspace.query_input_work is not None:
                q_work = (
                    q
                    if q_is_planned_work
                    else _copy_wave_to_work(
                        workspace.query_input_work,
                        q,
                        token_count=token_count,
                    )
                )
        if q_work is not None and not raw_query_input:
            if values is None or values.scale is None:
                raise RuntimeError("backward scale was not initialized")
            q_work.mul_(values.scale)
        pair_start = 1 + plan.head_dimension
        if plan.query_fold_impl == "triton_materialized":
            from .hd_block_gemm_feature_ops import fold_triton_query_feature_gradient

            d_q = None
            if need_q:
                if (
                    q is None
                    or q_work is None
                    or workspace.query_output is None
                    or values is None
                    or values.b is None
                    or values.c is None
                    or values.scale is None
                ):
                    raise RuntimeError("dQ workspace or values were not initialized")
                d_q = _wave_view(
                    workspace.query_output,
                    wave_shape=d_phi_q.shape[:-1],
                    token_count=token_count,
                    feature_dimension=plan.head_dimension,
                )
            coefficients = workspace.coefficients
            fold_triton_query_feature_gradient(
                q_work,
                d_phi_q,
                d_q,
                b=values.b if need_q and values is not None else None,
                c=values.c if need_q and values is not None else None,
                scale=values.scale if needs_input_q and values is not None else None,
                pair_rows=pairs.rows if pairs is not None else None,
                pair_columns=pairs.columns if pairs is not None else None,
                pair_multiplicity=pairs.multiplicity if pairs is not None else None,
                a_block_partials=(coefficients.a_block_partials if need_a else None),
                b_block_partials=(coefficients.b_block_partials if need_b else None),
                c_block_partials=(coefficients.c_block_partials if need_c else None),
                block_start=start if coefficient_requested else None,
                q_is_raw=raw_query_input,
                valid_tokens=q_token_count if raw_query_input else token_count,
                plan=plan,
            )
            if coefficient_requested:
                prepared.coefficient_coverage[start : start + wave_blocks] = [
                    True
                ] * wave_blocks
            return d_q, None, None, None

        generic_query_fold = workspace.generic_query_fold
        d_q = None
        if need_q:
            if (
                q is None
                or q_work is None
                or workspace.query_output is None
                or generic_query_fold is None
                or generic_query_fold.pair_source is None
                or values is None
                or values.b is None
                or values.c is None
                or values.scale is None
                or pairs is None
            ):
                raise RuntimeError("dQ workspace or values were not initialized")
            d_q = _wave_view(
                workspace.query_output,
                wave_shape=q.shape[:-1],
                token_count=token_count,
                feature_dimension=plan.head_dimension,
            )
            source = _wave_view(
                generic_query_fold.pair_source,
                wave_shape=q.shape[:-1],
                token_count=token_count,
                feature_dimension=plan.pair_count,
            )
            torch.mul(
                d_phi_q[..., 1:pair_start],
                _head_feature_view(values.b, wave_ndim=q.ndim),
                out=d_q,
            )
            d_q.mul_(values.scale)
            torch.index_select(q_work, -1, pairs.columns, out=source)
            source.mul_(d_phi_q[..., pair_start : pair_start + plan.pair_count])
            source.mul_(_head_feature_view(values.c, wave_ndim=q.ndim))
            source.mul_(_pair_metadata_view(pairs.multiplicity, wave_ndim=q.ndim))
            source.mul_(values.scale)
            _scatter_pair_source(d_q, pairs.rows, source)
            torch.index_select(q_work, -1, pairs.rows, out=source)
            source.mul_(d_phi_q[..., pair_start : pair_start + plan.pair_count])
            source.mul_(_head_feature_view(values.c, wave_ndim=q.ndim))
            source.mul_(_pair_metadata_view(pairs.multiplicity, wave_ndim=q.ndim))
            source.mul_(values.scale)
            _scatter_pair_source(d_q, pairs.columns, source)

        if coefficient_requested:
            partial_slice = slice(start, start + wave_blocks)
            coefficients = workspace.coefficients
            if need_a:
                if coefficients.a_block_partials is None:
                    raise RuntimeError("dA block partials were not initialized")
                torch.sum(
                    d_phi_q[..., 0],
                    dim=3,
                    dtype=torch.float32,
                    out=coefficients.a_block_partials[:, partial_slice].permute(
                        0, 2, 1
                    ),
                )
            if need_b:
                if (
                    q is None
                    or q_work is None
                    or generic_query_fold is None
                    or generic_query_fold.linear_source is None
                ):
                    raise RuntimeError("dB source workspace was not initialized")
                linear = _wave_view(
                    generic_query_fold.linear_source,
                    wave_shape=q.shape[:-1],
                    token_count=token_count,
                    feature_dimension=plan.head_dimension,
                )
                torch.mul(d_phi_q[..., 1:pair_start], q_work, out=linear)
                if coefficients.b_block_partials is None:
                    raise RuntimeError("dB block partials were not initialized")
                torch.sum(
                    linear,
                    dim=3,
                    dtype=torch.float32,
                    out=coefficients.b_block_partials[:, partial_slice].permute(
                        0, 2, 1, 3
                    ),
                )
            if need_c:
                if (
                    q is None
                    or q_work is None
                    or generic_query_fold is None
                    or generic_query_fold.pair_source is None
                    or generic_query_fold.pair_second is None
                    or pairs is None
                    or coefficients.c_block_partials is None
                ):
                    raise RuntimeError("dC workspace was not initialized")
                source = _wave_view(
                    generic_query_fold.pair_source,
                    wave_shape=q.shape[:-1],
                    token_count=token_count,
                    feature_dimension=plan.pair_count,
                )
                second = _wave_view(
                    generic_query_fold.pair_second,
                    wave_shape=q.shape[:-1],
                    token_count=token_count,
                    feature_dimension=plan.pair_count,
                )
                torch.index_select(q_work, -1, pairs.rows, out=source)
                torch.index_select(q_work, -1, pairs.columns, out=second)
                source.mul_(second)
                source.mul_(d_phi_q[..., pair_start : pair_start + plan.pair_count])
                source.mul_(_pair_metadata_view(pairs.multiplicity, wave_ndim=q.ndim))
                torch.sum(
                    source,
                    dim=3,
                    dtype=torch.float32,
                    out=coefficients.c_block_partials[:, partial_slice].permute(
                        0, 2, 1, 3
                    ),
                )
            prepared.coefficient_coverage[start : start + wave_blocks] = [
                True
            ] * wave_blocks
        return d_q, None, None, None
    except Exception:
        prepared._fail()
        raise


def _finalize_query_coefficient_gradients(
    prepared: _PreparedFeatureContext,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Reduce complete block partials in fixed batch-major/block-major order."""
    prepared = _require_prepared(prepared, mode=_BACKWARD)
    if prepared.coefficient_state != _COEFFICIENT_ACCUMULATING:
        raise RuntimeError("coefficient gradient accumulation is not active")
    coverage = prepared.coefficient_coverage
    if coverage is None or not all(coverage):
        raise RuntimeError(
            "coefficient gradient accumulation is incomplete; all blocks required"
        )
    workspace = prepared.backward_workspace
    if workspace is None:
        raise RuntimeError("backward coefficient workspace was not initialized")
    coefficients = workspace.coefficients
    plan = prepared.plan
    need_a, need_b, need_c = plan.requested_gradient_mask[3:]
    try:
        if need_a:
            if coefficients.a_block_partials is None or coefficients.a_output is None:
                raise RuntimeError("dA buffers were not initialized")
            torch.sum(
                coefficients.a_block_partials.view(-1, plan.query_heads),
                dim=0,
                dtype=torch.float32,
                out=coefficients.a_output,
            )
        if need_b:
            if coefficients.b_block_partials is None or coefficients.b_output is None:
                raise RuntimeError("dB buffers were not initialized")
            torch.sum(
                coefficients.b_block_partials.view(
                    -1, plan.query_heads, plan.head_dimension
                ),
                dim=0,
                dtype=torch.float32,
                out=coefficients.b_output,
            )
        if need_c:
            if coefficients.c_block_partials is None or coefficients.c_output is None:
                raise RuntimeError("dC buffers were not initialized")
            torch.sum(
                coefficients.c_block_partials.view(
                    -1, plan.query_heads, plan.pair_count
                ),
                dim=0,
                dtype=torch.float32,
                out=coefficients.c_output,
            )
        released_coefficients = replace(
            coefficients,
            a_block_partials=None,
            b_block_partials=None,
            c_block_partials=None,
        )
        released_workspace = replace(workspace, coefficients=released_coefficients)
    except Exception:
        prepared._fail()
        raise
    prepared.backward_workspace = released_workspace
    prepared.coefficient_state = _COEFFICIENT_FINALIZED
    return (
        released_coefficients.a_output if need_a else None,
        released_coefficients.b_output if need_b else None,
        released_coefficients.c_output if need_c else None,
    )


__all__: tuple[str, ...] = ()
