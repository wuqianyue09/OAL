"""Vectorized forward packed-feature transforms."""

from __future__ import annotations

import math

import torch

from .hd_block_gemm_feature_context import (
    _BACKWARD,
    _FORWARD,
    _PreparedFeatureContext,
    _copy_into,
    _copy_wave_to_work,
    _ensure_backward_workspace,
    _ensure_forward_workspace,
    _ensure_pairs,
    _ensure_prepared_values,
    _head_feature_view,
    _head_scalar_view,
    _pair_metadata_view,
    _require_feature_wave,
    _require_no_grad_tensors,
    _require_prepared,
    _require_raw_values,
    _wave_view,
)


def _require_feature_builder_context(
    prepared: object,
) -> _PreparedFeatureContext:
    if not isinstance(prepared, _PreparedFeatureContext):
        raise TypeError("prepared must be a prepared feature context")
    if prepared.mode not in (_FORWARD, _BACKWARD):
        raise ValueError("prepared context has an invalid feature-builder mode")
    return _require_prepared(prepared, mode=prepared.mode)


def _feature_builder_storage(
    prepared: _PreparedFeatureContext,
    *,
    key: bool,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    if prepared.mode == _FORWARD:
        _ensure_forward_workspace(prepared)
        workspace = prepared.forward_workspace
        if workspace is None:
            raise RuntimeError("forward feature workspace was not initialized")
        return (
            workspace.q_input_work,
            workspace.k_input_work,
            workspace.pair_work,
            workspace.pair_second_work,
            workspace.phi_k_output,
            workspace.phi_q_output,
        )
    _ensure_backward_workspace(
        prepared,
        feature_key=key,
        feature_query=not key,
    )
    workspace = prepared.backward_workspace
    if workspace is None:
        raise RuntimeError("backward feature workspace was not initialized")
    return (
        workspace.query_input_work,
        workspace.feature_k_input_work,
        workspace.feature_pair_work,
        workspace.feature_pair_second_work,
        workspace.phi_k_output,
        workspace.phi_q_output,
    )


def _contiguous_pair_work_view(
    storage: torch.Tensor,
    *,
    wave_shape: torch.Size,
    pair_count: int,
) -> torch.Tensor:
    """Take a contiguous logical prefix from reusable pair scratch.

    Slicing the planned batch/head/token axes can make an otherwise dense
    scratch view non-contiguous.  MPS ``index_select(out=...)`` requires a
    genuinely contiguous destination, so scratch ownership is expressed as a
    flat prefix before restoring the logical wave shape.
    """
    required = math.prod(wave_shape) * pair_count
    return storage.view(-1)[:required].view(*wave_shape, pair_count)


def _require_key_feature_output(
    output: object,
    *,
    key: torch.Tensor,
    token_count: int,
    plan: object,
) -> torch.Tensor:
    if not isinstance(output, torch.Tensor):
        raise TypeError("key feature output must be a tensor")
    expected_shape = (
        plan.batch_size,
        plan.key_value_heads,
        token_count,
        plan.physical_feature_dimension,
    )
    if tuple(output.shape) != expected_shape:
        raise ValueError(f"key feature output must have shape {expected_shape}")
    expected_dtype = getattr(torch, plan.feature_storage)
    if output.dtype != expected_dtype:
        raise ValueError(f"key feature output must use {expected_dtype}")
    if output.device != key.device:
        raise ValueError("key feature output must match the K device")
    if output.layout != torch.strided:
        raise ValueError("key feature output must use strided layout")
    if output.stride(-1) != 1 or output.stride(-2) != plan.physical_feature_dimension:
        raise ValueError(
            "key feature output must have a dense feature axis and token stride"
        )
    occupied_span = 1
    for stride, dimension in sorted(
        (stride, dimension)
        for dimension, stride in zip(output.shape, output.stride(), strict=True)
        if dimension > 1
    ):
        if stride < occupied_span:
            raise ValueError("key feature output must not have internal overlap")
        occupied_span += (dimension - 1) * stride
    maximum_offset = output.storage_offset() + sum(
        (dimension - 1) * stride
        for dimension, stride in zip(output.shape, output.stride(), strict=True)
    )
    storage_elements = output.untyped_storage().nbytes() // output.element_size()
    if maximum_offset >= storage_elements:
        raise ValueError("key feature output exceeds its backing storage")
    if key.untyped_storage().data_ptr() == output.untyped_storage().data_ptr():
        raise ValueError("K and key feature output must not overlap")
    return output


def _build_key_features(
    k: torch.Tensor,
    *,
    prepared: _PreparedFeatureContext,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build one whole planned key-feature wave in constant dispatches."""
    prepared = _require_feature_builder_context(prepared)
    raw_values = _require_raw_values(prepared)
    _require_no_grad_tensors(k, raw_values.a, raw_values.b, raw_values.c)
    plan = prepared.plan
    k, token_count = _require_feature_wave(
        k, name="K", head_count=plan.key_value_heads, plan=plan
    )
    _ensure_prepared_values(prepared)
    _ensure_pairs(prepared)
    (
        _,
        k_input_storage,
        pair_storage,
        pair_second_storage,
        phi_k_storage,
        _,
    ) = _feature_builder_storage(
        prepared,
        key=True,
    )
    pairs = prepared.pairs
    if pairs is None:
        raise RuntimeError("forward key workspace was not initialized")
    workspace_output = out is None
    if workspace_output:
        if phi_k_storage is None:
            raise RuntimeError("forward key workspace was not initialized")
        output = _wave_view(
            phi_k_storage,
            wave_shape=k.shape[:-1],
            token_count=token_count,
            feature_dimension=plan.physical_feature_dimension,
        )
    else:
        output = _require_key_feature_output(
            out,
            key=k,
            token_count=token_count,
            plan=plan,
        )
    if plan.feature_padding != "none":
        if workspace_output and phi_k_storage is not None:
            phi_k_storage[:, :, token_count:, :].zero_()
        if plan.key_feature_impl == "generic_materialized":
            output[..., plan.feature_dimension :].zero_()
    if plan.key_feature_impl == "triton_materialized":
        from .hd_block_gemm_key_ops import build_triton_key_features

        try:
            return build_triton_key_features(
                k,
                output,
                pair_rows=pairs.rows,
                pair_columns=pairs.columns,
                plan=plan,
            )
        except Exception:
            prepared._fail()
            raise
    if pair_storage is None:
        raise RuntimeError("forward key workspace was not initialized")
    try:
        pair_work = _contiguous_pair_work_view(
            pair_storage,
            wave_shape=k.shape[:-1],
            pair_count=plan.pair_count,
        )
        pair_start = 1 + plan.head_dimension
        pair_end = pair_start + plan.pair_count
        output[..., 0].fill_(1.0)
        linear = output[..., 1:pair_start]
        if plan.precision == "bf16_tensorcore":
            if k_input_storage is None or pair_second_storage is None:
                raise RuntimeError("Tensor-Core key workspace was not initialized")
            k_work = _copy_wave_to_work(
                k_input_storage,
                k,
                token_count=token_count,
            )
            _copy_into(linear, k_work)
            pair_second = _contiguous_pair_work_view(
                pair_second_storage,
                wave_shape=k.shape[:-1],
                pair_count=plan.pair_count,
            )
            pair_output = output[..., pair_start:pair_end]
            torch.index_select(k_work, -1, pairs.rows, out=pair_work)
            torch.index_select(k_work, -1, pairs.columns, out=pair_second)
            pair_work.mul_(pair_second)
            _copy_into(pair_output, pair_work)
            return output
        _copy_into(linear, k)
        pair_output = output[..., pair_start:pair_end]
        torch.index_select(linear, -1, pairs.rows, out=pair_work)
        _copy_into(pair_output, pair_work)
        torch.index_select(linear, -1, pairs.columns, out=pair_work)
        pair_output.mul_(pair_work)
        return output
    except Exception:
        prepared._fail()
        raise


def _build_query_features(
    q: torch.Tensor,
    *,
    prepared: _PreparedFeatureContext,
) -> torch.Tensor:
    """Build one whole weighted FP32 query-feature wave."""
    prepared = _require_feature_builder_context(prepared)
    raw_values = _require_raw_values(prepared)
    _require_no_grad_tensors(q, raw_values.a, raw_values.b, raw_values.c)
    plan = prepared.plan
    q, token_count = _require_feature_wave(
        q, name="Q", head_count=plan.query_heads, plan=plan
    )
    _ensure_prepared_values(prepared)
    _ensure_pairs(prepared)
    (
        q_input_storage,
        _,
        pair_storage,
        pair_second_storage,
        _,
        phi_q_storage,
    ) = _feature_builder_storage(
        prepared,
        key=False,
    )
    values = prepared.values
    pairs = prepared.pairs
    if (
        phi_q_storage is None
        or values is None
        or values.a is None
        or values.b is None
        or values.c is None
        or values.scale is None
        or pairs is None
    ):
        raise RuntimeError("forward query workspace was not initialized")
    output = _wave_view(
        phi_q_storage,
        wave_shape=q.shape[:-1],
        token_count=token_count,
        feature_dimension=plan.physical_feature_dimension,
    )
    if plan.feature_padding != "none":
        phi_q_storage[:, :, token_count:, :].zero_()
        if plan.query_feature_impl == "generic_materialized":
            output[..., plan.feature_dimension :].zero_()
    if plan.query_feature_impl == "triton_materialized":
        from .hd_block_gemm_feature_ops import build_triton_query_features

        try:
            return build_triton_query_features(
                q,
                output,
                a=values.a,
                b=values.b,
                c=values.c,
                scale=values.scale,
                pair_rows=pairs.rows,
                pair_columns=pairs.columns,
                pair_multiplicity=pairs.multiplicity,
                plan=plan,
            )
        except Exception:
            prepared._fail()
            raise
    if q_input_storage is None or pair_storage is None or phi_q_storage is None:
        raise RuntimeError("forward query workspace was not initialized")
    try:
        q_work = _copy_wave_to_work(q_input_storage, q, token_count=token_count)
        q_work.mul_(values.scale)
        pair_work = _contiguous_pair_work_view(
            pair_storage,
            wave_shape=q.shape[:-1],
            pair_count=plan.pair_count,
        )
        pair_start = 1 + plan.head_dimension
        pair_end = pair_start + plan.pair_count
        if plan.precision == "bf16_tensorcore":
            if pair_second_storage is None:
                raise RuntimeError("Tensor-Core query workspace was not initialized")
            _copy_into(
                output[..., 0],
                _head_scalar_view(values.a, wave_ndim=q.ndim),
            )
            linear_work = _contiguous_pair_work_view(
                pair_storage,
                wave_shape=q.shape[:-1],
                pair_count=plan.head_dimension,
            )
            torch.mul(
                q_work,
                _head_feature_view(values.b, wave_ndim=q.ndim),
                out=linear_work,
            )
            _copy_into(output[..., 1:pair_start], linear_work)
            pair_work = _contiguous_pair_work_view(
                pair_storage,
                wave_shape=q.shape[:-1],
                pair_count=plan.pair_count,
            )
            pair_second = _contiguous_pair_work_view(
                pair_second_storage,
                wave_shape=q.shape[:-1],
                pair_count=plan.pair_count,
            )
            torch.index_select(q_work, -1, pairs.rows, out=pair_work)
            torch.index_select(q_work, -1, pairs.columns, out=pair_second)
            pair_work.mul_(pair_second)
            pair_work.mul_(_head_feature_view(values.c, wave_ndim=q.ndim))
            pair_work.mul_(_pair_metadata_view(pairs.multiplicity, wave_ndim=q.ndim))
            _copy_into(output[..., pair_start:pair_end], pair_work)
            return output
        _copy_into(output[..., 0], _head_scalar_view(values.a, wave_ndim=q.ndim))
        torch.mul(
            q_work,
            _head_feature_view(values.b, wave_ndim=q.ndim),
            out=output[..., 1:pair_start],
        )
        pair_output = output[..., pair_start:pair_end]
        torch.index_select(q_work, -1, pairs.rows, out=pair_work)
        _copy_into(pair_output, pair_work)
        torch.index_select(q_work, -1, pairs.columns, out=pair_work)
        pair_output.mul_(pair_work)
        pair_output.mul_(_head_feature_view(values.c, wave_ndim=q.ndim))
        pair_output.mul_(_pair_metadata_view(pairs.multiplicity, wave_ndim=q.ndim))
        return output
    except Exception:
        prepared._fail()
        raise


__all__: tuple[str, ...] = ()
