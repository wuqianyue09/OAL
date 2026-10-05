"""Private analytic backward stages for the plan-bound HD Block-GEMM path."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch

from . import hd_block_gemm_normalization_ops
from .hd_block_gemm_backward_wave import (
    _compute_local_ds,
    _expand_kv_block_wave,
    _get_dt_wave,
    _get_g_wave as _get_staged_g_wave,
    _kv_wave_vjp,
    _prepare_wave_inputs,
    _query_wave_vjp,
    _reduce_gqa_block_wave,
    _stage_full_bf16_gradient,
)
from .hd_block_gemm_contractions import _bmm_fp32
from .hd_block_gemm_feature_context import (
    _PreparedFeatureContext,
    _ensure_backward_workspace,
    _ensure_pairs,
    _ensure_prepared_values,
    _planned_empty,
    _release_backward_feature_outputs,
    _require_prepared,
)
from .hd_block_gemm_feature_backward import (
    _begin_query_coefficient_gradient_accumulation,
    _finalize_query_coefficient_gradients,
    _fold_key_feature_gradient,
    _fold_query_feature_gradient,
)
from .hd_block_gemm_feature_forward import (
    _build_key_features,
    _build_query_features,
)
from .hd_block_gemm_profiling import (
    _QUERY_DS_STAGE,
    _SHARED_DS_STAGE,
    _SHARED_GQA_COPY_STAGE,
    _SHARED_K_FEATURE_STAGE,
    _record_hd_bmm_stage,
    _record_hd_contraction,
    _record_hd_stage,
)
from .hd_block_gemm_runtime import _require_eps


def _query_fold_q_wave(
    q: torch.Tensor,
    *,
    token_start: int,
    token_end: int,
    valid_tokens: int,
    capacity_tokens: int,
    wave_blocks: int,
    workspace: object | None,
    plan: object,
) -> tuple[torch.Tensor, bool]:
    """Select raw BF16 Q or construct the legacy scaled-work precursor."""
    if getattr(plan, "query_fold_input", "staged_fp32") == "raw":
        return q[:, :, token_start:token_end, :], False
    if workspace is None or workspace.query_input_work is None:
        raise RuntimeError("backward query input work was not initialized")
    q_work = workspace.query_input_work
    q_work[:, :, :valid_tokens, :].copy_(q[:, :, token_start:token_end, :])
    if valid_tokens < capacity_tokens:
        q_work[:, :, valid_tokens:, :].zero_()
    return (
        q_work.view(
            plan.batch_size,
            plan.query_heads,
            plan.feature_wave_blocks,
            plan.token_block,
            plan.head_dimension,
        )[:, :, :wave_blocks],
        True,
    )


def _allocate_planned(
    prepared: _PreparedFeatureContext,
    *,
    name: str,
) -> torch.Tensor:
    """Allocate one named buffer under an active backward invocation lease."""
    prepared = _require_prepared(prepared, mode="backward")
    return _planned_empty(prepared, name)


def _require_saved_tensor(
    tensor: object,
    *,
    name: str,
    shape: Sequence[int],
    prepared: _PreparedFeatureContext,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tensor.layout != torch.strided:
        raise ValueError(f"{name} must use strided layout")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} shape does not match the plan")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must use planned FP32 storage")
    if str(tensor.device) != prepared.plan.device:
        raise ValueError(f"{name} device does not match the plan")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must use planned contiguous storage")
    return tensor


def _require_saved_local_score(
    tensor: object,
    *,
    prepared: _PreparedFeatureContext,
) -> torch.Tensor | None:
    """Validate the optional forward-owned BF16 score cache exactly once."""
    plan = prepared.plan
    if not plan.save_local_score:
        if tensor is not None:
            raise ValueError("recompute plan must not receive a saved local score")
        return None
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("saved_local_score_tc must be a tensor")
    expected = plan.buffer("saved_local_score_tc")
    if (
        tensor.layout != torch.strided
        or tuple(tensor.shape) != expected.shape
        or tensor.dtype != torch.bfloat16
        or str(tensor.device) != plan.device
        or not tensor.is_contiguous()
    ):
        raise ValueError("saved local score does not match the plan")
    return tensor


def _require_saved_phi_k(
    tensor: object,
    *,
    prepared: _PreparedFeatureContext,
) -> torch.Tensor | None:
    """Validate one optional forward-owned BF16 key-feature cache."""
    plan = prepared.plan
    if plan.key_retention != "backward":
        if tensor is not None:
            raise ValueError("non-backward retention must not receive saved PhiK")
        return None
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("saved_phi_k_tc must be a tensor")
    expected = plan.buffer("saved_phi_k_tc")
    if (
        tensor.layout != torch.strided
        or tuple(tensor.shape) != expected.shape
        or tensor.dtype != torch.bfloat16
        or str(tensor.device) != plan.device
        or not tensor.is_contiguous()
    ):
        raise ValueError("saved PhiK does not match the plan")
    return tensor


def _require_upstream(
    gradient: object,
    *,
    name: str,
    shape: Sequence[int],
    dtype: torch.dtype,
    prepared: _PreparedFeatureContext,
) -> torch.Tensor | None:
    if gradient is None:
        return None
    if not isinstance(gradient, torch.Tensor):
        raise TypeError(f"{name} must be a tensor or None")
    if gradient.layout != torch.strided:
        raise ValueError(f"{name} must use strided layout")
    if tuple(gradient.shape) != tuple(shape):
        raise ValueError(f"{name} shape does not match the plan")
    if gradient.dtype != dtype:
        raise ValueError(f"{name} dtype does not match its forward output")
    if str(gradient.device) != prepared.plan.device:
        raise ValueError(f"{name} device does not match the plan")
    return gradient


def _augmented_normalization_vjp(
    grad_output: torch.Tensor | None,
    grad_numerator: torch.Tensor | None,
    grad_denominator: torch.Tensor | None,
    *,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    eps: float,
    prepared: _PreparedFeatureContext,
) -> torch.Tensor:
    """Return FP32 ``[dNumerator, dDenominator]`` for normalized output."""
    prepared = _require_prepared(prepared, mode="backward")
    plan = prepared.plan
    eps = _require_eps(eps)
    output_shape = (
        plan.batch_size,
        plan.query_heads,
        plan.sequence_length,
        plan.value_dimension,
    )
    denominator_shape = (*output_shape[:-1], 1)
    numerator = _require_saved_tensor(
        numerator,
        name="numerator",
        shape=output_shape,
        prepared=prepared,
    )
    denominator = _require_saved_tensor(
        denominator,
        name="denominator",
        shape=denominator_shape,
        prepared=prepared,
    )
    output_dtype = getattr(torch, plan.output_dtype)
    grad_output = _require_upstream(
        grad_output,
        name="grad_output",
        shape=output_shape,
        dtype=output_dtype,
        prepared=prepared,
    )
    grad_numerator = _require_upstream(
        grad_numerator,
        name="grad_numerator",
        shape=output_shape,
        dtype=torch.float32,
        prepared=prepared,
    )
    grad_denominator = _require_upstream(
        grad_denominator,
        name="grad_denominator",
        shape=denominator_shape,
        dtype=torch.float32,
        prepared=prepared,
    )

    augmented_gradient = _allocate_planned(
        prepared,
        name="backward_normalization_g",
    )
    if plan.backward_normalize_impl == "triton":
        with _record_hd_stage("hd.backward.normalize"):
            hd_block_gemm_normalization_ops.backward_normalize_triton(
                grad_output,
                grad_numerator,
                grad_denominator,
                numerator,
                denominator,
                augmented_gradient,
                eps=eps,
                plan=plan,
            )
        return augmented_gradient
    denominator_work = _allocate_planned(
        prepared,
        name="backward_normalization_denominator_work",
    )
    active = _allocate_planned(
        prepared,
        name="backward_normalization_active",
    )
    with _record_hd_stage("hd.backward.normalize"):
        denominator_work.copy_(denominator).clamp_min_(eps)
        torch.ge(denominator, eps, out=active)

        value_gradient = augmented_gradient[..., : plan.value_dimension]
        denominator_gradient = augmented_gradient[..., plan.value_dimension :]
        if grad_output is None:
            augmented_gradient.zero_()
        else:
            value_gradient.copy_(grad_output)
            if plan.precision == "bf16_tensorcore":
                value_gradient.mul_(numerator)
                torch.sum(
                    value_gradient,
                    dim=-1,
                    keepdim=True,
                    dtype=torch.float32,
                    out=denominator_gradient,
                )
            else:
                row_count = plan.batch_size * plan.query_heads * plan.sequence_length
                torch.bmm(
                    value_gradient.view(row_count, 1, plan.value_dimension),
                    numerator.view(row_count, plan.value_dimension, 1),
                    out=denominator_gradient.view(row_count, 1, 1),
                )
            denominator_gradient.div_(denominator_work).div_(denominator_work)
            denominator_gradient.neg_().mul_(active)
            if plan.precision == "bf16_tensorcore":
                value_gradient.copy_(grad_output)
            value_gradient.div_(denominator_work)

        if grad_numerator is not None:
            value_gradient.add_(grad_numerator)
        if grad_denominator is not None:
            denominator_gradient.add_(grad_denominator)
    return augmented_gradient


def _stage_backward_gradient(
    augmented_gradient: torch.Tensor,
    *,
    prepared: _PreparedFeatureContext,
) -> torch.Tensor | None:
    """Materialize the optional invocation-local BF16 gradient cache once."""
    prepared = _require_prepared(prepared, mode="backward")
    if prepared.plan.gradient_staging == "per_wave":
        return None
    if prepared.plan.gradient_staging != "full_bf16":
        raise RuntimeError("unsupported prepared gradient staging policy")
    cache = _allocate_planned(prepared, name="backward_g_full_bf16")
    with _record_hd_stage("hd.backward.g_stage_full_bf16"):
        return _stage_full_bf16_gradient(
            augmented_gradient,
            cache=cache,
            token_block=prepared.plan.token_block,
            feature_wave_blocks=prepared.plan.feature_wave_blocks,
        )


def _require_plan_input(
    tensor: object,
    *,
    name: Literal["Q", "K", "V"],
    prepared: _PreparedFeatureContext,
) -> torch.Tensor:
    """Validate one Q/K/V input against the plan's canonical geometry."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tensor.layout != torch.strided:
        raise ValueError(f"{name} must use strided layout")
    plan = prepared.plan
    shape = (
        plan.batch_size,
        plan.query_heads if name == "Q" else plan.key_value_heads,
        plan.sequence_length,
        plan.value_dimension if name == "V" else plan.head_dimension,
    )
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} shape does not match the plan")
    if tensor.dtype != getattr(torch, prepared.plan.input_dtype):
        raise ValueError(f"{name} dtype does not match the plan")
    if str(tensor.device) != prepared.plan.device:
        raise ValueError(f"{name} device does not match the plan")
    return tensor


def _get_key_wave(
    k: torch.Tensor,
    *,
    block_start: int,
    wave_index: int,
    wave_blocks: int,
    saved_phi_k_tc: torch.Tensor | None,
    prepared: _PreparedFeatureContext,
) -> tuple[torch.Tensor, int]:
    """Return one Hkv-owned key-feature wave from saved storage or its builder."""
    plan = prepared.plan
    token_start = block_start * plan.token_block
    token_end = min(
        plan.sequence_length,
        token_start + wave_blocks * plan.token_block,
    )
    valid_tokens = token_end - token_start
    if saved_phi_k_tc is not None:
        return saved_phi_k_tc[wave_index], valid_tokens
    _build_key_features(
        k[:, :, token_start:token_end, :],
        prepared=prepared,
    )
    workspace = prepared.backward_workspace
    if workspace is None or workspace.phi_k_output is None:
        raise RuntimeError("backward key feature workspace was not initialized")
    phi_k_blocks = workspace.phi_k_output.view(
        plan.batch_size,
        plan.key_value_heads,
        plan.feature_wave_blocks,
        plan.token_block,
        plan.physical_feature_dimension,
    )
    capacity_tokens = plan.feature_wave_blocks * plan.token_block
    if valid_tokens < capacity_tokens:
        phi_k_blocks.view(
            plan.batch_size,
            plan.key_value_heads,
            capacity_tokens,
            plan.physical_feature_dimension,
        )[:, :, valid_tokens:, :].zero_()
    return phi_k_blocks, valid_tokens


def _copy_backward_key_value_wave(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_start: int,
    wave_index: int,
    wave_blocks: int,
    u_wave: torch.Tensor,
    saved_phi_k_tc: torch.Tensor | None,
    prepared: _PreparedFeatureContext,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    phi_k_blocks, key_valid_tokens = _get_key_wave(
        k,
        block_start=block_start,
        wave_index=wave_index,
        wave_blocks=wave_blocks,
        saved_phi_k_tc=saved_phi_k_tc,
        prepared=prepared,
    )
    u_blocks, valid_tokens = _copy_backward_value_wave(
        v,
        block_start=block_start,
        wave_blocks=wave_blocks,
        u_wave=u_wave,
        prepared=prepared,
    )
    if key_valid_tokens != valid_tokens:
        raise RuntimeError("backward key/value wave token counts do not match")
    return (
        phi_k_blocks,
        u_blocks,
        valid_tokens,
    )


def _copy_backward_value_wave(
    v: torch.Tensor,
    *,
    block_start: int,
    wave_blocks: int,
    u_wave: torch.Tensor,
    prepared: _PreparedFeatureContext,
) -> tuple[torch.Tensor, int]:
    plan = prepared.plan
    token_start = block_start * plan.token_block
    token_end = min(
        plan.sequence_length,
        token_start + wave_blocks * plan.token_block,
    )
    valid_tokens = _prepare_wave_inputs(
        None,
        v,
        token_start=token_start,
        token_end=token_end,
        g_wave=None,
        u_wave=u_wave,
        value_dimension=plan.value_dimension,
    )
    return (
        u_wave.view(
            plan.batch_size,
            plan.key_value_heads,
            plan.feature_wave_blocks,
            plan.token_block,
            plan.augmented_value_dimension,
        ),
        valid_tokens,
    )


def _query_side_vjp(
    q: torch.Tensor | None,
    k: torch.Tensor,
    v: torch.Tensor,
    augmented_gradient: torch.Tensor,
    carry: torch.Tensor,
    *,
    full_bf16_gradient: torch.Tensor | None = None,
    saved_phi_k_tc: torch.Tensor | None = None,
    prepared: _PreparedFeatureContext,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Compute dQ/dA/dB/dC wavewise without mutating saved carry."""
    prepared = _require_prepared(prepared, mode="backward")
    plan = prepared.plan
    need_q, _, need_v, need_a, need_b, need_c = plan.requested_gradient_mask
    if not (need_q or need_a or need_b or need_c):
        return None, None, None, None
    needs_q_input = need_q or need_b or need_c
    q_for_fold = None
    if needs_q_input:
        q_for_fold = _require_plan_input(
            q,
            name="Q",
            prepared=prepared,
        )
    k = _require_plan_input(
        k,
        name="K",
        prepared=prepared,
    )
    v = _require_plan_input(
        v,
        name="V",
        prepared=prepared,
    )
    augmented_gradient = _require_saved_tensor(
        augmented_gradient,
        name="augmented_gradient",
        shape=(
            plan.batch_size,
            plan.query_heads,
            plan.sequence_length,
            plan.augmented_value_dimension,
        ),
        prepared=prepared,
    )
    carry = _require_saved_tensor(
        carry,
        name="carry",
        shape=(
            plan.batch_size,
            plan.key_value_heads,
            plan.number_blocks,
            plan.physical_feature_dimension,
            plan.augmented_value_dimension,
        ),
        prepared=prepared,
    )
    saved_phi_k_tc = _require_saved_phi_k(
        saved_phi_k_tc,
        prepared=prepared,
    )

    grad_q = _allocate_planned(prepared, name="grad_q") if need_q else None
    u_wave = _allocate_planned(prepared, name="backward_u_wave")
    g_wave = (
        _allocate_planned(prepared, name="backward_g_wave")
        if full_bf16_gradient is None
        else None
    )
    phi_k_query_wave = (
        _allocate_planned(prepared, name="backward_phi_k_query_wave")
        if plan.gqa_ratio > 1
        else None
    )
    u_query_wave = (
        _allocate_planned(prepared, name="backward_u_query_wave")
        if plan.gqa_ratio > 1
        else None
    )
    carry_query_wave = _allocate_planned(
        prepared,
        name="backward_carry_query_wave",
    )
    ds_local = _allocate_planned(prepared, name="backward_ds_local")
    ds_local_tc = (
        _allocate_planned(prepared, name="backward_ds_local_tc")
        if plan.precision == "bf16_tensorcore"
        else None
    )
    d_phi_q = _allocate_planned(prepared, name="backward_dphi_q")
    d_phi_q_local = (
        _allocate_planned(prepared, name="backward_dphi_q_local")
        if plan.precision == "bf16_tensorcore"
        else None
    )
    coefficient_requested = need_a or need_b or need_c
    if coefficient_requested:
        _begin_query_coefficient_gradient_accumulation(prepared)
    _ensure_backward_workspace(prepared, query_fold=True)
    _ensure_prepared_values(prepared)

    capacity_tokens = plan.feature_wave_blocks * plan.token_block
    for wave_index, block_start in enumerate(
        range(
            0,
            plan.number_blocks,
            plan.feature_wave_blocks,
        )
    ):
        wave_blocks = min(
            plan.feature_wave_blocks,
            plan.number_blocks - block_start,
        )
        with _record_hd_stage("hd.backward.k_feature_query"):
            phi_k, u, valid_tokens = _copy_backward_key_value_wave(
                k,
                v,
                block_start=block_start,
                wave_index=wave_index,
                wave_blocks=wave_blocks,
                u_wave=u_wave,
                saved_phi_k_tc=saved_phi_k_tc,
                prepared=prepared,
            )
        token_start = block_start * plan.token_block
        token_end = token_start + valid_tokens
        current_g_wave = _get_staged_g_wave(
            augmented_gradient,
            token_start=token_start,
            token_end=token_end,
            wave_index=wave_index,
            g_wave=g_wave,
            full_bf16_cache=full_bf16_gradient,
        )

        phi_k_for_query = phi_k
        u_for_query = u
        if plan.gqa_ratio > 1:
            if phi_k_query_wave is None or u_query_wave is None:
                raise RuntimeError("GQA query-wave storage was not initialized")
            phi_k_for_query = _expand_kv_block_wave(
                phi_k,
                phi_k_query_wave,
                gqa_ratio=plan.gqa_ratio,
            )
            u_for_query = _expand_kv_block_wave(
                u,
                u_query_wave,
                gqa_ratio=plan.gqa_ratio,
            )

        carry_query_blocks = carry_query_wave.view(
            plan.batch_size,
            plan.key_value_heads,
            plan.gqa_ratio,
            plan.feature_wave_blocks,
            plan.physical_feature_dimension,
            plan.augmented_value_dimension,
        )
        if wave_blocks < plan.feature_wave_blocks:
            carry_query_blocks[:, :, :, wave_blocks:, :, :].zero_()
        carry_query_blocks[:, :, :, :wave_blocks, :, :].copy_(
            carry[:, :, block_start : block_start + wave_blocks].unsqueeze(2)
        )

        flat_batches = plan.batch_size * plan.query_heads * plan.feature_wave_blocks
        g_3d = current_g_wave.view(
            flat_batches,
            plan.token_block,
            plan.augmented_value_dimension,
        )
        u_3d = u_for_query.view(
            flat_batches,
            plan.token_block,
            plan.augmented_value_dimension,
        )
        ds_3d = ds_local.view(
            flat_batches,
            plan.token_block,
            plan.token_block,
        )
        ds_tc_3d = (
            ds_local_tc.view(
                flat_batches,
                plan.token_block,
                plan.token_block,
            )
            if ds_local_tc is not None
            else None
        )
        if d_phi_q is None:
            raise RuntimeError("materialized query fold dPhiQ was not initialized")
        d_phi_q_3d = d_phi_q.view(
            flat_batches,
            plan.token_block,
            plan.physical_feature_dimension,
        )
        phi_k_3d = phi_k_for_query.view(
            flat_batches,
            plan.token_block,
            plan.physical_feature_dimension,
        )
        local_3d = (
            d_phi_q_local.view(
                flat_batches,
                plan.token_block,
                plan.physical_feature_dimension,
            )
            if d_phi_q_local is not None
            else None
        )
        _query_wave_vjp(
            g_3d,
            u_3d,
            phi_k_3d,
            carry_query_wave.view(
                flat_batches,
                plan.physical_feature_dimension,
                plan.augmented_value_dimension,
            ),
            ds=ds_3d,
            ds_tc=ds_tc_3d,
            d_phi_q=d_phi_q_3d,
            d_phi_q_local=local_3d,
            precomputed_ds=None,
            precision=plan.precision,
            backend_identity=plan.contraction_backend_identity,
            loaded_backend_token=prepared.contraction_backend_token,
            bmm_fp32=_bmm_fp32,
            record_stage=_record_hd_stage,
        )

        d_phi_q_blocks = d_phi_q.view(
            plan.batch_size,
            plan.query_heads,
            plan.feature_wave_blocks,
            plan.token_block,
            plan.physical_feature_dimension,
        )[:, :, :wave_blocks]
        workspace = prepared.backward_workspace
        q_work_blocks = None
        q_is_planned_work = False
        if needs_q_input:
            if q_for_fold is None:
                raise RuntimeError("backward query input was not initialized")
            q_work_blocks, q_is_planned_work = _query_fold_q_wave(
                q_for_fold,
                token_start=token_start,
                token_end=token_end,
                valid_tokens=valid_tokens,
                capacity_tokens=capacity_tokens,
                wave_blocks=wave_blocks,
                workspace=workspace,
                plan=plan,
            )
        with _record_hd_stage("hd.backward.query_fold"):
            d_q_wave, _, _, _ = _fold_query_feature_gradient(
                q_work_blocks,
                d_phi_q_blocks,
                prepared=prepared,
                block_start=block_start if coefficient_requested else None,
                q_is_planned_work=q_is_planned_work,
            )
        if grad_q is not None:
            if d_q_wave is None:
                raise RuntimeError("requested dQ wave was not produced")
            grad_q[:, :, token_start:token_end, :].copy_(
                d_q_wave.view(
                    plan.batch_size,
                    plan.query_heads,
                    wave_blocks * plan.token_block,
                    plan.head_dimension,
                )[:, :, :valid_tokens, :]
            )

    if coefficient_requested:
        with _record_hd_stage("hd.backward.coefficient_reduce"):
            grad_a, grad_b, grad_c = _finalize_query_coefficient_gradients(prepared)
    else:
        grad_a = grad_b = grad_c = None
    if not need_v:
        _release_backward_feature_outputs(prepared, key=True)
    return grad_q, grad_a, grad_b, grad_c


def _query_dcarry(
    q: torch.Tensor,
    augmented_gradient: torch.Tensor,
    *,
    full_bf16_gradient: torch.Tensor | None = None,
    prepared: _PreparedFeatureContext,
) -> torch.Tensor:
    """Build KV-owned block dCarry with one bounded query-head reduction."""
    prepared = _require_prepared(prepared, mode="backward")
    plan = prepared.plan
    if not (plan.requested_gradient_mask[1] or plan.requested_gradient_mask[2]):
        raise RuntimeError("dCarry requires a requested K or V gradient")
    q = _require_plan_input(
        q,
        name="Q",
        prepared=prepared,
    )
    augmented_gradient = _require_saved_tensor(
        augmented_gradient,
        name="augmented_gradient",
        shape=(
            plan.batch_size,
            plan.query_heads,
            plan.sequence_length,
            plan.augmented_value_dimension,
        ),
        prepared=prepared,
    )
    dcarry = _allocate_planned(prepared, name="backward_dcarry")
    g_wave = (
        _allocate_planned(prepared, name="backward_g_wave")
        if full_bf16_gradient is None
        else None
    )
    query_partials = _allocate_planned(
        prepared,
        name="backward_dcarry_query_wave",
    )
    capacity_tokens = plan.feature_wave_blocks * plan.token_block

    for wave_index, block_start in enumerate(
        range(0, plan.number_blocks, plan.feature_wave_blocks)
    ):
        wave_blocks = min(
            plan.feature_wave_blocks,
            plan.number_blocks - block_start,
        )
        token_start = block_start * plan.token_block
        token_end = min(
            plan.sequence_length,
            token_start + wave_blocks * plan.token_block,
        )
        valid_tokens = token_end - token_start
        with _record_hd_stage("hd.backward.q_feature_dcarry"):
            _build_query_features(
                q[:, :, token_start:token_end, :],
                prepared=prepared,
            )
        workspace = prepared.backward_workspace
        if workspace is None or workspace.phi_q_output is None:
            raise RuntimeError("backward query feature workspace was not initialized")
        phi_q_wave = workspace.phi_q_output
        if valid_tokens < capacity_tokens:
            phi_q_wave[:, :, valid_tokens:, :].zero_()
        current_g_wave = _get_staged_g_wave(
            augmented_gradient,
            token_start=token_start,
            token_end=token_end,
            wave_index=wave_index,
            g_wave=g_wave,
            full_bf16_cache=full_bf16_gradient,
        )

        flat_batches = plan.batch_size * plan.query_heads * plan.feature_wave_blocks
        with _record_hd_stage("hd.backward.dcarry"):
            _bmm_fp32(
                phi_q_wave.view(
                    flat_batches,
                    plan.token_block,
                    plan.physical_feature_dimension,
                ).transpose(1, 2),
                current_g_wave.view(
                    flat_batches,
                    plan.token_block,
                    plan.augmented_value_dimension,
                ),
                out=query_partials.view(
                    flat_batches,
                    plan.physical_feature_dimension,
                    plan.augmented_value_dimension,
                ),
                precision=plan.precision,
                backend_identity=plan.contraction_backend_identity,
                loaded_backend_token=prepared.contraction_backend_token,
            )
        query_blocks = query_partials.view(
            plan.batch_size,
            plan.key_value_heads,
            plan.gqa_ratio,
            plan.feature_wave_blocks,
            plan.physical_feature_dimension,
            plan.augmented_value_dimension,
        )
        destination = dcarry[:, :, block_start : block_start + wave_blocks, :, :]
        if plan.gqa_ratio == 1:
            destination.copy_(query_blocks[:, :, 0, :wave_blocks])
        else:
            torch.sum(
                query_blocks[:, :, :, :wave_blocks],
                dim=2,
                dtype=torch.float32,
                out=destination,
            )
    return dcarry


def _exclusive_right_scan(
    dcarry: torch.Tensor,
    *,
    prepared: _PreparedFeatureContext,
) -> torch.Tensor:
    """Return dT[b] = sum(dCarry[c] for c>b) with one full-NB scan."""
    prepared = _require_prepared(prepared, mode="backward")
    plan = prepared.plan
    if not (plan.requested_gradient_mask[1] or plan.requested_gradient_mask[2]):
        raise RuntimeError("right scan requires a requested K or V gradient")
    dcarry = _require_saved_tensor(
        dcarry,
        name="dcarry",
        shape=(
            plan.batch_size,
            plan.key_value_heads,
            plan.number_blocks,
            plan.physical_feature_dimension,
            plan.augmented_value_dimension,
        ),
        prepared=prepared,
    )
    dt = _allocate_planned(prepared, name="backward_right_scan_dt")
    total = _allocate_planned(prepared, name="backward_right_scan_total")
    with _record_hd_stage("hd.backward.right_scan"):
        torch.cumsum(dcarry, dim=2, out=dt)
        total.copy_(dt[:, :, -1:, :, :])
        dt.neg_().add_(total)
    return dt


def _add_kv_cross_term_(
    destination: torch.Tensor,
    left: torch.Tensor,
    dt: torch.Tensor,
    *,
    block_start: int,
    wave_blocks: int,
    transpose_dt: bool,
    precision: str,
    scratch: torch.Tensor | None,
    prepared: _PreparedFeatureContext,
) -> None:
    """Add a cross term using a batch axis supported by the contraction backend."""
    plan = prepared.plan
    batch_size, head_count = destination.shape[:2]
    if precision == "bf16_tensorcore" and scratch is None:
        raise RuntimeError("Tensor-Core KV cross scratch was not initialized")
    scratch_blocks = None
    if scratch is not None:
        if scratch.numel() != destination.numel():
            raise ValueError("KV cross scratch does not match its destination")
        scratch_blocks = scratch.view_as(destination)
    if (
        plan.kv_cross_impl == "batched_dense"
        and precision == "bf16_tensorcore"
        and wave_blocks == plan.feature_wave_blocks
    ):
        left_full = left[:, :, :wave_blocks]
        right_full = dt[:, :, block_start : block_start + wave_blocks]
        if transpose_dt:
            right_full = right_full.transpose(-2, -1)
        output_full = destination[:, :, :wave_blocks]
        if scratch_blocks is None:
            raise RuntimeError("Tensor-Core KV cross scratch was not initialized")
        cross_full = scratch_blocks[:, :, :wave_blocks]
        flattened_batch = batch_size * head_count * wave_blocks
        try:
            left_batched = left_full.view(flattened_batch, *left_full.shape[-2:])
            right_batched = right_full.view(flattened_batch, *right_full.shape[-2:])
            output_batched = output_full.view(flattened_batch, *output_full.shape[-2:])
            cross_batched = cross_full.view(flattened_batch, *cross_full.shape[-2:])
        except RuntimeError:
            pass
        else:
            with _record_hd_stage("hd.backward.kv_cross"):
                _bmm_fp32(
                    left_batched,
                    right_batched,
                    out=cross_batched,
                    precision=precision,
                    backend_identity=plan.contraction_backend_identity,
                    loaded_backend_token=prepared.contraction_backend_token,
                )
            output_batched.add_(cross_batched)
            return
    # BF16's cuBLAS contract requires tightly packed matrices on the BMM
    # batch axis. Fixing a wave block and flattening batch/head leaves a
    # wave-sized gap between matrices (notably Hkv=8 with waves of 2 or 4).
    # Batch over consecutive blocks within each head instead, including tails.
    # FP32 torch.baddbmm also accepts strided batches and keeps the old choice.
    if precision == "bf16_tensorcore" or batch_size * head_count <= wave_blocks:
        for batch in range(batch_size):
            for head in range(head_count):
                right = dt[
                    batch,
                    head,
                    block_start : block_start + wave_blocks,
                ]
                if transpose_dt:
                    right = right.transpose(1, 2)
                output = destination[batch, head, :wave_blocks]
                if precision == "bf16_tensorcore":
                    if scratch_blocks is None:
                        raise RuntimeError(
                            "Tensor-Core KV cross scratch was not initialized"
                        )
                    cross = scratch_blocks[batch, head, :wave_blocks]
                    with _record_hd_stage("hd.backward.kv_cross"):
                        _bmm_fp32(
                            left[batch, head, :wave_blocks],
                            right,
                            out=cross,
                            precision=precision,
                            backend_identity=plan.contraction_backend_identity,
                            loaded_backend_token=prepared.contraction_backend_token,
                        )
                    output.add_(cross)
                else:
                    left_blocks = left[batch, head, :wave_blocks]
                    with _record_hd_stage("hd.backward.kv_cross"):
                        _record_hd_contraction(left_blocks, right, output)
                        with _record_hd_bmm_stage(left_blocks, right, output):
                            torch.baddbmm(
                                output,
                                left_blocks,
                                right,
                                beta=1.0,
                                out=output,
                            )
        return

    # BF16 returns through the dense/head-batched paths above.
    for wave_block in range(wave_blocks):
        right = dt[:, :, block_start + wave_block].flatten(0, 1)
        if transpose_dt:
            right = right.transpose(1, 2)
        output = destination[:, :, wave_block].flatten(0, 1)
        left_block = left[:, :, wave_block].flatten(0, 1)
        with _record_hd_stage("hd.backward.kv_cross"):
            _record_hd_contraction(left_block, right, output)
            with _record_hd_bmm_stage(left_block, right, output):
                torch.baddbmm(
                    output,
                    left_block,
                    right,
                    beta=1.0,
                    out=output,
                )


def _kv_side_vjp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor | None,
    augmented_gradient: torch.Tensor,
    dt: torch.Tensor,
    *,
    full_bf16_gradient: torch.Tensor | None = None,
    saved_local_score_tc: torch.Tensor | None = None,
    saved_phi_k_tc: torch.Tensor | None = None,
    prepared: _PreparedFeatureContext,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Compute requested KV gradients wavewise from local and carried terms."""
    prepared = _require_prepared(prepared, mode="backward")
    plan = prepared.plan
    _, need_k, need_v, _, _, _ = plan.requested_gradient_mask
    if not (need_k or need_v):
        return None, None
    q = _require_plan_input(
        q,
        name="Q",
        prepared=prepared,
    )
    k = _require_plan_input(
        k,
        name="K",
        prepared=prepared,
    )
    v_for_k = None
    if need_k:
        v_for_k = _require_plan_input(
            v,
            name="V",
            prepared=prepared,
        )
    augmented_gradient = _require_saved_tensor(
        augmented_gradient,
        name="augmented_gradient",
        shape=(
            plan.batch_size,
            plan.query_heads,
            plan.sequence_length,
            plan.augmented_value_dimension,
        ),
        prepared=prepared,
    )
    dt = _require_saved_tensor(
        dt,
        name="dt",
        shape=(
            plan.batch_size,
            plan.key_value_heads,
            plan.number_blocks,
            plan.physical_feature_dimension,
            plan.augmented_value_dimension,
        ),
        prepared=prepared,
    )
    saved_local_score_tc = _require_saved_local_score(
        saved_local_score_tc,
        prepared=prepared,
    )
    saved_phi_k_tc = _require_saved_phi_k(
        saved_phi_k_tc,
        prepared=prepared,
    )

    grad_k = _allocate_planned(prepared, name="grad_k") if need_k else None
    grad_v = _allocate_planned(prepared, name="grad_v") if need_v else None
    g_wave = (
        _allocate_planned(prepared, name="backward_g_wave")
        if full_bf16_gradient is None
        else None
    )
    u_wave = _allocate_planned(prepared, name="backward_u_wave") if need_k else None
    ds_local = _allocate_planned(prepared, name="backward_ds_local") if need_k else None
    ds_local_tc = (
        _allocate_planned(prepared, name="backward_ds_local_tc")
        if need_k and plan.precision == "bf16_tensorcore"
        else None
    )
    d_phi_k = _allocate_planned(prepared, name="backward_dphi_k") if need_k else None
    d_phi_k_cross = (
        _allocate_planned(prepared, name="backward_dphi_k_cross")
        if need_k and plan.precision == "bf16_tensorcore"
        else None
    )
    d_phi_k_query = (
        _allocate_planned(prepared, name="backward_dphi_k_query_wave")
        if need_k and plan.gqa_ratio > 1
        else None
    )
    u_query_wave = (
        _allocate_planned(prepared, name="backward_u_query_wave")
        if need_k and plan.gqa_ratio > 1
        else None
    )
    local_score = (
        _allocate_planned(prepared, name="backward_local_score")
        if need_v and saved_local_score_tc is None
        else None
    )
    local_score_tc = (
        _allocate_planned(prepared, name="backward_local_score_tc")
        if (
            need_v
            and plan.precision == "bf16_tensorcore"
            and saved_local_score_tc is None
        )
        else None
    )
    du = _allocate_planned(prepared, name="backward_du") if need_v else None
    du_cross = (
        _allocate_planned(prepared, name="backward_du_cross")
        if need_v and plan.precision == "bf16_tensorcore"
        else None
    )
    du_query = (
        _allocate_planned(prepared, name="backward_du_query_wave")
        if need_v and plan.gqa_ratio > 1
        else None
    )
    phi_k_query_wave = (
        _allocate_planned(prepared, name="backward_phi_k_query_wave")
        if need_v and plan.gqa_ratio > 1
        else None
    )
    dt_wave_tc = (
        _allocate_planned(prepared, name="backward_dt_wave_tc")
        if plan.precision == "bf16_tensorcore"
        else None
    )
    capacity_tokens = plan.feature_wave_blocks * plan.token_block

    for wave_index, block_start in enumerate(
        range(0, plan.number_blocks, plan.feature_wave_blocks)
    ):
        wave_blocks = min(
            plan.feature_wave_blocks,
            plan.number_blocks - block_start,
        )
        token_start = block_start * plan.token_block
        token_end = min(
            plan.sequence_length,
            token_start + wave_blocks * plan.token_block,
        )
        valid_tokens = token_end - token_start
        dt_for_cross, cross_block_start = _get_dt_wave(
            dt,
            block_start=block_start,
            wave_blocks=wave_blocks,
            dt_wave_tc=dt_wave_tc,
            precision=plan.precision,
        )
        with _record_hd_stage("hd.backward.q_feature_kv"):
            _build_query_features(
                q[:, :, token_start:token_end, :],
                prepared=prepared,
            )
        workspace = prepared.backward_workspace
        if workspace is None or workspace.phi_q_output is None:
            raise RuntimeError("backward query feature workspace was not initialized")
        phi_q_wave = workspace.phi_q_output
        if valid_tokens < capacity_tokens:
            phi_q_wave[:, :, valid_tokens:, :].zero_()
        current_g_wave = _get_staged_g_wave(
            augmented_gradient,
            token_start=token_start,
            token_end=token_end,
            wave_index=wave_index,
            g_wave=g_wave,
            full_bf16_cache=full_bf16_gradient,
        )

        phi_q_blocks = phi_q_wave.view(
            plan.batch_size,
            plan.query_heads,
            plan.feature_wave_blocks,
            plan.token_block,
            plan.physical_feature_dimension,
        )
        g_blocks = current_g_wave.view(
            plan.batch_size,
            plan.query_heads,
            plan.feature_wave_blocks,
            plan.token_block,
            plan.augmented_value_dimension,
        )
        query_block_batches = (
            plan.batch_size * plan.query_heads * plan.feature_wave_blocks
        )
        phi_q_3d = phi_q_blocks.view(
            query_block_batches,
            plan.token_block,
            plan.physical_feature_dimension,
        )
        g_3d = g_blocks.view(
            query_block_batches,
            plan.token_block,
            plan.augmented_value_dimension,
        )

        phi_k_blocks = None
        if need_v:
            with _record_hd_stage("hd.backward.k_feature_kv"):
                phi_k_blocks, key_valid_tokens = _get_key_wave(
                    k,
                    block_start=block_start,
                    wave_index=wave_index,
                    wave_blocks=wave_blocks,
                    saved_phi_k_tc=saved_phi_k_tc,
                    prepared=prepared,
                )
            if key_valid_tokens != valid_tokens:
                raise RuntimeError("backward key wave has an invalid token count")

        if need_k:
            if v_for_k is None or u_wave is None or ds_local is None or d_phi_k is None:
                raise RuntimeError("dK wave storage was not initialized")
            u_blocks, copied_tokens = _copy_backward_value_wave(
                v_for_k,
                block_start=block_start,
                wave_blocks=wave_blocks,
                u_wave=u_wave,
                prepared=prepared,
            )
            if copied_tokens != valid_tokens:
                raise RuntimeError("backward value wave copied an invalid token count")
            u_for_query = u_blocks
            if plan.gqa_ratio > 1:
                if u_query_wave is None:
                    raise RuntimeError("GQA value-wave storage was not initialized")
                u_for_query = _expand_kv_block_wave(
                    u_blocks,
                    u_query_wave,
                    gqa_ratio=plan.gqa_ratio,
                )

            ds_3d = ds_local.view(
                query_block_batches,
                plan.token_block,
                plan.token_block,
            )
            ds_tc_3d = (
                ds_local_tc.view(
                    query_block_batches,
                    plan.token_block,
                    plan.token_block,
                )
                if ds_local_tc is not None
                else None
            )
            local_dphi_output = d_phi_k
            if plan.gqa_ratio > 1:
                if d_phi_k_query is None:
                    raise RuntimeError("GQA dPhiK storage was not initialized")
                local_dphi_output = d_phi_k_query
            _kv_wave_vjp(
                phi_q_3d,
                g_3d,
                u=u_for_query.view(
                    query_block_batches,
                    plan.token_block,
                    plan.augmented_value_dimension,
                ),
                phi_k=None,
                ds=ds_3d,
                ds_tc=ds_tc_3d,
                d_phi_k=local_dphi_output.view(
                    query_block_batches,
                    plan.token_block,
                    plan.physical_feature_dimension,
                ),
                local_score=None,
                local_score_tc=None,
                du=None,
                precomputed_ds=None,
                precision=plan.precision,
                backend_identity=plan.contraction_backend_identity,
                loaded_backend_token=prepared.contraction_backend_token,
                bmm_fp32=_bmm_fp32,
                record_stage=_record_hd_stage,
            )
            d_phi_k_blocks = d_phi_k.view(
                plan.batch_size,
                plan.key_value_heads,
                plan.feature_wave_blocks,
                plan.token_block,
                plan.physical_feature_dimension,
            )
            if plan.gqa_ratio > 1:
                _reduce_gqa_block_wave(
                    d_phi_k_query,
                    d_phi_k_blocks,
                    gqa_ratio=plan.gqa_ratio,
                )
            _add_kv_cross_term_(
                d_phi_k_blocks,
                u_blocks,
                dt_for_cross,
                block_start=cross_block_start,
                wave_blocks=wave_blocks,
                transpose_dt=True,
                precision=plan.precision,
                scratch=d_phi_k_cross,
                prepared=prepared,
            )
            with _record_hd_stage("hd.backward.kv_fold"):
                d_k_wave = _fold_key_feature_gradient(
                    k[:, :, token_start:token_end, :],
                    d_phi_k[:, :, :valid_tokens, :],
                    prepared=prepared,
                )
            if d_k_wave is None or grad_k is None:
                raise RuntimeError("requested dK wave was not produced")
            grad_k[:, :, token_start:token_end, :].copy_(d_k_wave)

        if need_v:
            if (
                phi_k_blocks is None
                or (local_score is None and saved_local_score_tc is None)
                or du is None
                or grad_v is None
            ):
                raise RuntimeError("dV wave storage was not initialized")
            phi_k_for_query = phi_k_blocks
            if plan.gqa_ratio > 1:
                if phi_k_query_wave is None:
                    raise RuntimeError("GQA key-feature storage was not initialized")
                phi_k_for_query = _expand_kv_block_wave(
                    phi_k_blocks,
                    phi_k_query_wave,
                    gqa_ratio=plan.gqa_ratio,
                )
            local_score_3d = (
                local_score.view(
                    query_block_batches,
                    plan.token_block,
                    plan.token_block,
                )
                if local_score is not None
                else None
            )
            local_score_tc_3d = (
                local_score_tc.view(
                    query_block_batches,
                    plan.token_block,
                    plan.token_block,
                )
                if local_score_tc is not None
                else None
            )
            local_du_output = du
            if plan.gqa_ratio > 1:
                if du_query is None:
                    raise RuntimeError("GQA dU storage was not initialized")
                local_du_output = du_query
            _kv_wave_vjp(
                phi_q_3d,
                g_3d,
                u=None,
                phi_k=phi_k_for_query.view(
                    query_block_batches,
                    plan.token_block,
                    plan.physical_feature_dimension,
                ),
                ds=None,
                ds_tc=None,
                d_phi_k=None,
                local_score=local_score_3d,
                local_score_tc=local_score_tc_3d,
                saved_local_score_tc=(
                    saved_local_score_tc[wave_index].view(
                        query_block_batches,
                        plan.token_block,
                        plan.token_block,
                    )
                    if saved_local_score_tc is not None
                    else None
                ),
                du=local_du_output.view(
                    query_block_batches,
                    plan.token_block,
                    plan.augmented_value_dimension,
                ),
                precomputed_ds=None,
                precision=plan.precision,
                backend_identity=plan.contraction_backend_identity,
                loaded_backend_token=prepared.contraction_backend_token,
                bmm_fp32=_bmm_fp32,
                record_stage=_record_hd_stage,
            )
            du_blocks = du.view(
                plan.batch_size,
                plan.key_value_heads,
                plan.feature_wave_blocks,
                plan.token_block,
                plan.augmented_value_dimension,
            )
            if plan.gqa_ratio > 1:
                _reduce_gqa_block_wave(
                    du_query,
                    du_blocks,
                    gqa_ratio=plan.gqa_ratio,
                )
            _add_kv_cross_term_(
                du_blocks,
                phi_k_blocks,
                dt_for_cross,
                block_start=cross_block_start,
                wave_blocks=wave_blocks,
                transpose_dt=False,
                precision=plan.precision,
                scratch=du_cross,
                prepared=prepared,
            )
            grad_v[:, :, token_start:token_end, :].copy_(
                du[:, :, :valid_tokens, : plan.value_dimension]
            )

    _release_backward_feature_outputs(prepared, key=True, query=True)
    return grad_k, grad_v


def _shared_wave_vjp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    augmented_gradient: torch.Tensor,
    carry: torch.Tensor,
    dt: torch.Tensor,
    *,
    full_bf16_gradient: torch.Tensor | None = None,
    saved_local_score_tc: torch.Tensor | None = None,
    saved_phi_k_tc: torch.Tensor | None = None,
    prepared: _PreparedFeatureContext,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Consume query and KV gradients from one shared K/U/g/dS wave."""
    prepared = _require_prepared(prepared, mode="backward")
    plan = prepared.plan
    if plan.backward_schedule != "shared_wave":
        raise RuntimeError("shared backward requires a shared_wave plan")
    need_q, need_k, need_v, need_a, need_b, need_c = plan.requested_gradient_mask
    query_side = need_q or need_a or need_b or need_c
    if not query_side or not (need_k or need_v):
        raise RuntimeError("shared backward requires query and KV consumers")
    if plan.query_gradient_flow != "materialized" or plan.query_consumer_stages:
        raise RuntimeError("shared backward does not support OP-3/OP-4 query flows")

    q = _require_plan_input(
        q,
        name="Q",
        prepared=prepared,
    )
    k = _require_plan_input(
        k,
        name="K",
        prepared=prepared,
    )
    v = _require_plan_input(
        v,
        name="V",
        prepared=prepared,
    )
    augmented_gradient = _require_saved_tensor(
        augmented_gradient,
        name="augmented_gradient",
        shape=(
            plan.batch_size,
            plan.query_heads,
            plan.sequence_length,
            plan.augmented_value_dimension,
        ),
        prepared=prepared,
    )
    carry = _require_saved_tensor(
        carry,
        name="carry",
        shape=(
            plan.batch_size,
            plan.key_value_heads,
            plan.number_blocks,
            plan.physical_feature_dimension,
            plan.augmented_value_dimension,
        ),
        prepared=prepared,
    )
    dt = _require_saved_tensor(
        dt,
        name="dt",
        shape=(
            plan.batch_size,
            plan.key_value_heads,
            plan.number_blocks,
            plan.physical_feature_dimension,
            plan.augmented_value_dimension,
        ),
        prepared=prepared,
    )
    saved_local_score_tc = _require_saved_local_score(
        saved_local_score_tc,
        prepared=prepared,
    )
    saved_phi_k_tc = _require_saved_phi_k(
        saved_phi_k_tc,
        prepared=prepared,
    )

    grad_q = _allocate_planned(prepared, name="grad_q") if need_q else None
    grad_k = _allocate_planned(prepared, name="grad_k") if need_k else None
    grad_v = _allocate_planned(prepared, name="grad_v") if need_v else None
    u_wave = _allocate_planned(prepared, name="backward_u_wave")
    g_wave = (
        _allocate_planned(prepared, name="backward_g_wave")
        if full_bf16_gradient is None
        else None
    )
    phi_k_query_wave = (
        _allocate_planned(prepared, name="backward_phi_k_query_wave")
        if plan.gqa_ratio > 1
        else None
    )
    u_query_wave = (
        _allocate_planned(prepared, name="backward_u_query_wave")
        if plan.gqa_ratio > 1
        else None
    )
    carry_query_wave = _allocate_planned(
        prepared,
        name="backward_carry_query_wave",
    )
    ds_local = _allocate_planned(prepared, name="backward_ds_local")
    ds_local_tc = (
        _allocate_planned(prepared, name="backward_ds_local_tc")
        if plan.precision == "bf16_tensorcore"
        else None
    )
    d_phi_q = _allocate_planned(prepared, name="backward_dphi_q")
    d_phi_q_local = (
        _allocate_planned(prepared, name="backward_dphi_q_local")
        if plan.precision == "bf16_tensorcore"
        else None
    )
    d_phi_k = _allocate_planned(prepared, name="backward_dphi_k") if need_k else None
    d_phi_k_cross = (
        _allocate_planned(prepared, name="backward_dphi_k_cross")
        if need_k and plan.precision == "bf16_tensorcore"
        else None
    )
    d_phi_k_query = (
        _allocate_planned(prepared, name="backward_dphi_k_query_wave")
        if need_k and plan.gqa_ratio > 1
        else None
    )
    local_score = (
        _allocate_planned(prepared, name="backward_local_score")
        if need_v and saved_local_score_tc is None
        else None
    )
    local_score_tc = (
        _allocate_planned(prepared, name="backward_local_score_tc")
        if (
            need_v
            and plan.precision == "bf16_tensorcore"
            and saved_local_score_tc is None
        )
        else None
    )
    du = _allocate_planned(prepared, name="backward_du") if need_v else None
    du_cross = (
        _allocate_planned(prepared, name="backward_du_cross")
        if need_v and plan.precision == "bf16_tensorcore"
        else None
    )
    du_query = (
        _allocate_planned(prepared, name="backward_du_query_wave")
        if need_v and plan.gqa_ratio > 1
        else None
    )
    dt_wave_tc = (
        _allocate_planned(prepared, name="backward_dt_wave_tc")
        if plan.precision == "bf16_tensorcore"
        else None
    )
    coefficient_requested = need_a or need_b or need_c
    if coefficient_requested:
        _begin_query_coefficient_gradient_accumulation(prepared)
    _ensure_backward_workspace(prepared, query_fold=True)
    _ensure_prepared_values(prepared)

    capacity_tokens = plan.feature_wave_blocks * plan.token_block
    flat_batches = plan.batch_size * plan.query_heads * plan.feature_wave_blocks
    for wave_index, block_start in enumerate(
        range(0, plan.number_blocks, plan.feature_wave_blocks)
    ):
        wave_blocks = min(
            plan.feature_wave_blocks,
            plan.number_blocks - block_start,
        )
        with _record_hd_stage(_SHARED_K_FEATURE_STAGE):
            phi_k, u_blocks, valid_tokens = _copy_backward_key_value_wave(
                k,
                v,
                block_start=block_start,
                wave_index=wave_index,
                wave_blocks=wave_blocks,
                u_wave=u_wave,
                saved_phi_k_tc=saved_phi_k_tc,
                prepared=prepared,
            )
        token_start = block_start * plan.token_block
        token_end = token_start + valid_tokens
        current_g_wave = _get_staged_g_wave(
            augmented_gradient,
            token_start=token_start,
            token_end=token_end,
            wave_index=wave_index,
            g_wave=g_wave,
            full_bf16_cache=full_bf16_gradient,
        )

        phi_k_for_query = phi_k
        u_for_query = u_blocks
        with _record_hd_stage(_SHARED_GQA_COPY_STAGE):
            if plan.gqa_ratio > 1:
                if phi_k_query_wave is None or u_query_wave is None:
                    raise RuntimeError("shared GQA storage was not initialized")
                phi_k_for_query = _expand_kv_block_wave(
                    phi_k,
                    phi_k_query_wave,
                    gqa_ratio=plan.gqa_ratio,
                )
                u_for_query = _expand_kv_block_wave(
                    u_blocks,
                    u_query_wave,
                    gqa_ratio=plan.gqa_ratio,
                )

            carry_query_blocks = carry_query_wave.view(
                plan.batch_size,
                plan.key_value_heads,
                plan.gqa_ratio,
                plan.feature_wave_blocks,
                plan.physical_feature_dimension,
                plan.augmented_value_dimension,
            )
            if wave_blocks < plan.feature_wave_blocks:
                carry_query_blocks[:, :, :, wave_blocks:, :, :].zero_()
            carry_query_blocks[:, :, :, :wave_blocks, :, :].copy_(
                carry[:, :, block_start : block_start + wave_blocks].unsqueeze(2)
            )

        g_3d = current_g_wave.view(
            flat_batches,
            plan.token_block,
            plan.augmented_value_dimension,
        )
        u_3d = u_for_query.view(
            flat_batches,
            plan.token_block,
            plan.augmented_value_dimension,
        )
        ds_3d = ds_local.view(
            flat_batches,
            plan.token_block,
            plan.token_block,
        )
        ds_tc_3d = (
            ds_local_tc.view(
                flat_batches,
                plan.token_block,
                plan.token_block,
            )
            if ds_local_tc is not None
            else None
        )
        ds_operand_3d = _compute_local_ds(
            g_3d,
            u_3d,
            ds=ds_3d,
            ds_tc=ds_tc_3d,
            precision=plan.precision,
            backend_identity=plan.contraction_backend_identity,
            loaded_backend_token=prepared.contraction_backend_token,
            stage=_SHARED_DS_STAGE,
            bmm_fp32=_bmm_fp32,
            record_stage=_record_hd_stage,
        )

        d_phi_q_3d = d_phi_q.view(
            flat_batches,
            plan.token_block,
            plan.physical_feature_dimension,
        )
        d_phi_q_local_3d = (
            d_phi_q_local.view(
                flat_batches,
                plan.token_block,
                plan.physical_feature_dimension,
            )
            if d_phi_q_local is not None
            else None
        )
        _query_wave_vjp(
            g_3d,
            u_3d,
            phi_k_for_query.view(
                flat_batches,
                plan.token_block,
                plan.physical_feature_dimension,
            ),
            carry_query_wave.view(
                flat_batches,
                plan.physical_feature_dimension,
                plan.augmented_value_dimension,
            ),
            ds=ds_3d,
            ds_tc=ds_tc_3d,
            d_phi_q=d_phi_q_3d,
            d_phi_q_local=d_phi_q_local_3d,
            precomputed_ds=ds_operand_3d,
            precision=plan.precision,
            backend_identity=plan.contraction_backend_identity,
            loaded_backend_token=prepared.contraction_backend_token,
            bmm_fp32=_bmm_fp32,
            record_stage=_record_hd_stage,
        )
        d_phi_q_blocks = d_phi_q.view(
            plan.batch_size,
            plan.query_heads,
            plan.feature_wave_blocks,
            plan.token_block,
            plan.physical_feature_dimension,
        )[:, :, :wave_blocks]
        workspace = prepared.backward_workspace
        q_work_blocks = None
        q_is_planned_work = False
        if need_q or need_b or need_c:
            q_work_blocks, q_is_planned_work = _query_fold_q_wave(
                q,
                token_start=token_start,
                token_end=token_end,
                valid_tokens=valid_tokens,
                capacity_tokens=capacity_tokens,
                wave_blocks=wave_blocks,
                workspace=workspace,
                plan=plan,
            )
        with _record_hd_stage("hd.backward.query_fold"):
            d_q_wave, _, _, _ = _fold_query_feature_gradient(
                q_work_blocks,
                d_phi_q_blocks,
                prepared=prepared,
                block_start=block_start if coefficient_requested else None,
                q_is_planned_work=q_is_planned_work,
            )
        if grad_q is not None:
            if d_q_wave is None:
                raise RuntimeError("requested shared dQ wave was not produced")
            grad_q[:, :, token_start:token_end, :].copy_(
                d_q_wave.view(
                    plan.batch_size,
                    plan.query_heads,
                    wave_blocks * plan.token_block,
                    plan.head_dimension,
                )[:, :, :valid_tokens, :]
            )

        dt_for_cross, cross_block_start = _get_dt_wave(
            dt,
            block_start=block_start,
            wave_blocks=wave_blocks,
            dt_wave_tc=dt_wave_tc,
            precision=plan.precision,
        )

        with _record_hd_stage("hd.backward.q_feature_kv"):
            _build_query_features(
                q[:, :, token_start:token_end, :],
                prepared=prepared,
            )
        workspace = prepared.backward_workspace
        if workspace is None or workspace.phi_q_output is None:
            raise RuntimeError("shared query feature workspace was not initialized")
        phi_q_wave = workspace.phi_q_output
        if valid_tokens < capacity_tokens:
            phi_q_wave[:, :, valid_tokens:, :].zero_()
        phi_q_3d = phi_q_wave.view(
            flat_batches,
            plan.token_block,
            plan.physical_feature_dimension,
        )

        if need_k:
            if d_phi_k is None or grad_k is None:
                raise RuntimeError("shared dK storage was not initialized")
            local_dphi_output = d_phi_k
            if plan.gqa_ratio > 1:
                if d_phi_k_query is None:
                    raise RuntimeError("shared GQA dPhiK storage was not initialized")
                local_dphi_output = d_phi_k_query
            _kv_wave_vjp(
                phi_q_3d,
                g_3d,
                u=u_3d,
                phi_k=None,
                ds=ds_3d,
                ds_tc=ds_tc_3d,
                d_phi_k=local_dphi_output.view(
                    flat_batches,
                    plan.token_block,
                    plan.physical_feature_dimension,
                ),
                local_score=None,
                local_score_tc=None,
                du=None,
                precomputed_ds=ds_operand_3d,
                precision=plan.precision,
                backend_identity=plan.contraction_backend_identity,
                loaded_backend_token=prepared.contraction_backend_token,
                bmm_fp32=_bmm_fp32,
                record_stage=_record_hd_stage,
            )
            d_phi_k_blocks = d_phi_k.view(
                plan.batch_size,
                plan.key_value_heads,
                plan.feature_wave_blocks,
                plan.token_block,
                plan.physical_feature_dimension,
            )
            if plan.gqa_ratio > 1:
                _reduce_gqa_block_wave(
                    d_phi_k_query,
                    d_phi_k_blocks,
                    gqa_ratio=plan.gqa_ratio,
                )
            _add_kv_cross_term_(
                d_phi_k_blocks,
                u_blocks,
                dt_for_cross,
                block_start=cross_block_start,
                wave_blocks=wave_blocks,
                transpose_dt=True,
                precision=plan.precision,
                scratch=d_phi_k_cross,
                prepared=prepared,
            )
            with _record_hd_stage("hd.backward.kv_fold"):
                d_k_wave = _fold_key_feature_gradient(
                    k[:, :, token_start:token_end, :],
                    d_phi_k[:, :, :valid_tokens, :],
                    prepared=prepared,
                )
            if d_k_wave is None:
                raise RuntimeError("requested shared dK wave was not produced")
            grad_k[:, :, token_start:token_end, :].copy_(d_k_wave)

        if need_v:
            if (
                (local_score is None and saved_local_score_tc is None)
                or du is None
                or grad_v is None
            ):
                raise RuntimeError("shared dV storage was not initialized")
            local_du_output = du
            if plan.gqa_ratio > 1:
                if du_query is None:
                    raise RuntimeError("shared GQA dU storage was not initialized")
                local_du_output = du_query
            _kv_wave_vjp(
                phi_q_3d,
                g_3d,
                u=None,
                phi_k=phi_k_for_query.view(
                    flat_batches,
                    plan.token_block,
                    plan.physical_feature_dimension,
                ),
                ds=None,
                ds_tc=None,
                d_phi_k=None,
                local_score=(
                    local_score.view(
                        flat_batches,
                        plan.token_block,
                        plan.token_block,
                    )
                    if local_score is not None
                    else None
                ),
                local_score_tc=(
                    local_score_tc.view(
                        flat_batches,
                        plan.token_block,
                        plan.token_block,
                    )
                    if local_score_tc is not None
                    else None
                ),
                saved_local_score_tc=(
                    saved_local_score_tc[wave_index].view(
                        flat_batches,
                        plan.token_block,
                        plan.token_block,
                    )
                    if saved_local_score_tc is not None
                    else None
                ),
                du=local_du_output.view(
                    flat_batches,
                    plan.token_block,
                    plan.augmented_value_dimension,
                ),
                precomputed_ds=None,
                precision=plan.precision,
                backend_identity=plan.contraction_backend_identity,
                loaded_backend_token=prepared.contraction_backend_token,
                bmm_fp32=_bmm_fp32,
                record_stage=_record_hd_stage,
            )
            du_blocks = du.view(
                plan.batch_size,
                plan.key_value_heads,
                plan.feature_wave_blocks,
                plan.token_block,
                plan.augmented_value_dimension,
            )
            if plan.gqa_ratio > 1:
                _reduce_gqa_block_wave(
                    du_query,
                    du_blocks,
                    gqa_ratio=plan.gqa_ratio,
                )
            _add_kv_cross_term_(
                du_blocks,
                phi_k,
                dt_for_cross,
                block_start=cross_block_start,
                wave_blocks=wave_blocks,
                transpose_dt=False,
                precision=plan.precision,
                scratch=du_cross,
                prepared=prepared,
            )
            grad_v[:, :, token_start:token_end, :].copy_(
                du[:, :, :valid_tokens, : plan.value_dimension]
            )

    if coefficient_requested:
        with _record_hd_stage("hd.backward.coefficient_reduce"):
            grad_a, grad_b, grad_c = _finalize_query_coefficient_gradients(prepared)
    else:
        grad_a = grad_b = grad_c = None
    _release_backward_feature_outputs(prepared, key=True, query=True)
    return grad_q, grad_k, grad_v, grad_a, grad_b, grad_c


__all__: tuple[str, ...] = ()
