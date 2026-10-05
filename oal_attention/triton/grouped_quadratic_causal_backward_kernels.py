"""CUDA kernels for the KV-owned grouped causal reverse suffix scan.

The correctness-first suffix scan lives in
``grouped_quadratic_causal_backward``.  This module only owns the bounded
Triton schedule for its ``dK``/``dV`` half: A0/A1 use token blocks, A2 uses
the complete packed-pair grid, and every fan-in is reduced through an explicit
fixed-order partial buffer.  No query-head or pair contribution is merged with
atomics.
"""

from __future__ import annotations

import torch
from typing import Final

from .grouped_quadratic_causal_common import (
    KernelPlan,
    planned_grouped_causal_scan_config,
)
from .grouped_quadratic_production_witness import active_production_witness_session

try:
    import triton
    import triton.language as tl
except (ImportError, ModuleNotFoundError):
    triton = None
    tl = None


def triton_is_available() -> bool:
    """Return whether the dedicated grouped causal suffix kernels are usable."""
    return triton is not None and tl is not None


_SUFFIX_PRODUCTION_WITNESS_CAPTURE_NAMES: Final[tuple[str, ...]] = (
    "suffix.a_inclusive",
)


def suffix_production_witness_capture_names() -> tuple[str, ...]:
    """Return the scalar probe emitted from the actual final A-suffix state."""
    return _SUFFIX_PRODUCTION_WITNESS_CAPTURE_NAMES


def _capture_suffix_production_witness(
    a2: torch.Tensor,
    *,
    plan: KernelPlan,
) -> None:
    session = active_production_witness_session()
    if session is None or not session.expects("suffix.a_inclusive"):
        return
    probe = session.request.probe_for("suffix")
    geometry = plan.geometry
    if (
        probe.batch_index >= geometry.B
        or probe.key_value_head >= geometry.Hkv
        or probe.query_head >= geometry.Hq
        or probe.token_index != 0
        or probe.pair_index >= geometry.pair_count
        or probe.channel_index >= geometry.augmented_value_dimension
        or probe.query_head // (geometry.Hq // geometry.Hkv) != probe.key_value_head
    ):
        raise ValueError(
            "suffix production witness probe is not a reverse-inclusive GQA coordinate"
        )
    flat_key_value_head = probe.batch_index * geometry.Hkv + probe.key_value_head
    session.capture_device_tensor(
        "suffix.a_inclusive",
        a2[flat_key_value_head, probe.pair_index, probe.channel_index].reshape(1),
    )


def _torch_dtype(dtype_name: str) -> torch.dtype:
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"unsupported planned tensor dtype: {dtype_name}")
    return dtype


def suffix_workspace_allocation_names(plan: KernelPlan) -> tuple[str, ...]:
    """Return physical records; pair row/column aliases are runtime-only views."""
    if not isinstance(plan, KernelPlan):
        raise TypeError("planned grouped causal suffix requires a KernelPlan")
    _, needs_k, needs_v, _, _, _ = plan.geometry.requested_gradient_mask
    if not needs_k and not needs_v:
        return ()
    names = ["suffix_state", "suffix_pair_metadata"]
    if needs_v:
        names.extend(("dv", "suffix_dv_macro", "suffix_dv_pair_partial"))
    if needs_k:
        names.extend(("dk", "suffix_dk_value_partial", "suffix_dk_pair_partial"))
    for name in names:
        allocation = plan.workspace.allocation(name)
        if allocation.alias_of is not None:
            raise ValueError(f"suffix workspace allocation {name} cannot be an alias")
    return tuple(names)


def allocate_planned_suffix_workspace(
    plan: KernelPlan,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Allocate physical records and expose pair-row/column runtime alias views."""
    buffers: dict[str, torch.Tensor] = {}
    names = suffix_workspace_allocation_names(plan)
    for name in names:
        if name == "suffix_pair_metadata":
            continue
        allocation = plan.workspace.allocation(name)
        kwargs = {"device": device, "dtype": _torch_dtype(allocation.dtype)}
        buffers[name] = (
            torch.zeros(allocation.shape, **kwargs)
            if name == "suffix_state"
            else torch.empty(allocation.shape, **kwargs)
        )
    if "suffix_pair_metadata" in names:
        allocation = plan.workspace.allocation("suffix_pair_metadata")
        pair_metadata = torch.tril_indices(
            plan.geometry.D,
            plan.geometry.D,
            device=device,
            dtype=torch.int32,
        )
        if tuple(
            pair_metadata.shape
        ) != allocation.shape or pair_metadata.dtype != _torch_dtype(allocation.dtype):
            raise AssertionError("canonical suffix pair metadata does not match plan")
        for name in ("suffix_pair_rows", "suffix_pair_columns"):
            alias = plan.workspace.allocation(name)
            if alias.alias_of != "suffix_pair_metadata":
                raise ValueError(f"{name} must alias suffix_pair_metadata")
        buffers["suffix_pair_metadata"] = pair_metadata
        # These are views of the exact torch.tril_indices allocation: no copy,
        # contiguous conversion, or second metadata allocation is permitted.
        buffers["suffix_pair_rows"] = pair_metadata[0]
        buffers["suffix_pair_columns"] = pair_metadata[1]
    return buffers


def _suffix_state_views(
    plan: KernelPlan,
    workspace: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create H0/H1/H2-shaped A views of one planned FP32 suffix state set."""
    state = workspace["suffix_state"]
    geometry = plan.geometry
    expected_shape = (
        geometry.B,
        geometry.Hkv,
        geometry.augmented_value_dimension,
        1 + geometry.D + geometry.pair_count,
    )
    if state.dtype != torch.float32 or tuple(state.shape) != expected_shape:
        raise ValueError("suffix_state does not match the exact FP32 plan record")
    for name in ("suffix_h0_view", "suffix_h1_view", "suffix_h2_view"):
        if plan.workspace.allocation(name).alias_of != "suffix_state":
            raise ValueError(f"{name} must remain a suffix_state alias")
    block_stride = state.stride(1)
    channel_stride = state.stride(2)
    feature_stride = state.stride(3)
    batch_key_value_heads = geometry.B * geometry.Hkv
    a0 = state.as_strided(
        (batch_key_value_heads, geometry.augmented_value_dimension),
        (block_stride, channel_stride),
    )
    a1 = state.as_strided(
        (batch_key_value_heads, geometry.D, geometry.augmented_value_dimension),
        (block_stride, feature_stride, channel_stride),
        storage_offset=1,
    )
    a2 = state.as_strided(
        (
            batch_key_value_heads,
            geometry.pair_count,
            geometry.augmented_value_dimension,
        ),
        (block_stride, feature_stride, channel_stride),
        storage_offset=1 + geometry.D,
    )
    return a0, a1, a2


def _flatten_suffix_kv_workspace(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    """Expose a zero-copy ``[B * Hkv, ...]`` view for KV-owned Triton grids."""
    if not isinstance(tensor, torch.Tensor) or tensor.ndim < 3:
        raise ValueError(f"{name} must be a rank-3-or-higher planned suffix buffer")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must remain a contiguous planned suffix buffer")
    return tensor.as_strided(
        (tensor.shape[0] * tensor.shape[1], *tensor.shape[2:]),
        (tensor.stride(1), *tensor.stride()[2:]),
    )


if triton_is_available():

    @triton.jit
    def _reverse_a0_a1_token_block_kernel(
        q_pointer,
        k_pointer,
        v_pointer,
        dim_groups_pointer,
        constant_pointer,
        linear_pointer,
        gradient_pointer,
        a0_pointer,
        a1_pointer,
        d_v_accumulator_pointer,
        d_k_partials_pointer,
        chunk_start,
        chunk_length,
        query_heads,
        key_value_heads,
        head_dimension,
        value_dimension,
        scale,
        stride_q_batch,
        stride_q_head,
        stride_q_token,
        stride_q_feature,
        stride_k_batch,
        stride_k_head,
        stride_k_token,
        stride_k_feature,
        stride_v_batch,
        stride_v_head,
        stride_v_token,
        stride_v_value,
        stride_gradient_batch,
        stride_gradient_head,
        stride_gradient_token,
        stride_gradient_value,
        stride_constant_head,
        stride_a0_batch,
        stride_a0_channel,
        stride_a1_batch,
        stride_a1_feature,
        stride_a1_channel,
        stride_d_v_batch,
        stride_d_v_token,
        stride_d_k_batch,
        stride_d_k_token,
        stride_d_k_value_tile,
        GROUPS_PER_HEAD: tl.constexpr,
        GMAX: tl.constexpr,
        D_PAD: tl.constexpr,
        GQA: tl.constexpr,
        BT: tl.constexpr,
        BV: tl.constexpr,
        WRITE_K: tl.constexpr,
        WRITE_V: tl.constexpr,
    ):
        value_tile = tl.program_id(0)
        batch_key_value_head = tl.program_id(1)
        batch_key_value_head_64 = batch_key_value_head.to(tl.int64)
        batch = batch_key_value_head // key_value_heads
        key_value_head = batch_key_value_head % key_value_heads
        values = value_tile * BV + tl.arange(0, BV)
        features = tl.arange(0, D_PAD)
        value_mask = values < value_dimension
        augmented_value_mask = values < value_dimension + 1
        feature_mask = features < head_dimension

        if WRITE_V:
            a0 = tl.load(
                a0_pointer
                + batch_key_value_head_64 * stride_a0_batch
                + values * stride_a0_channel,
                mask=augmented_value_mask,
                other=0.0,
            ).to(tl.float32)
        else:
            a0 = tl.zeros((BV,), dtype=tl.float32)
        a1 = tl.load(
            a1_pointer
            + batch_key_value_head_64 * stride_a1_batch
            + features[:, None] * stride_a1_feature
            + values[None, :] * stride_a1_channel,
            mask=feature_mask[:, None] & augmented_value_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        for reverse_step in tl.range(0, BT):
            active = reverse_step < chunk_length
            local_token = chunk_length - 1 - reverse_step
            token = chunk_start + local_token
            key = tl.load(
                k_pointer
                + batch.to(tl.int64) * stride_k_batch
                + key_value_head.to(tl.int64) * stride_k_head
                + token.to(tl.int64) * stride_k_token
                + features * stride_k_feature,
                mask=active & feature_mask,
                other=0.0,
            ).to(tl.float32)
            loaded_value = tl.load(
                v_pointer
                + batch.to(tl.int64) * stride_v_batch
                + key_value_head.to(tl.int64) * stride_v_head
                + token.to(tl.int64) * stride_v_token
                + values * stride_v_value,
                mask=active & value_mask,
                other=0.0,
            ).to(tl.float32)
            augmented_value = tl.where(
                active & (values == value_dimension), 1.0, loaded_value
            )

            for local_query in range(0, GQA):
                query_head = key_value_head * GQA + local_query
                gradient = tl.load(
                    gradient_pointer
                    + batch.to(tl.int64) * stride_gradient_batch
                    + query_head.to(tl.int64) * stride_gradient_head
                    + token.to(tl.int64) * stride_gradient_token
                    + values * stride_gradient_value,
                    mask=active & augmented_value_mask,
                    other=0.0,
                ).to(tl.float32)
                query = tl.load(
                    q_pointer
                    + batch.to(tl.int64) * stride_q_batch
                    + query_head.to(tl.int64) * stride_q_head
                    + token.to(tl.int64) * stride_q_token
                    + features * stride_q_feature,
                    mask=active & feature_mask,
                    other=0.0,
                ).to(tl.float32)
                if GROUPS_PER_HEAD:
                    groups = tl.load(
                        dim_groups_pointer
                        + query_head.to(tl.int64) * head_dimension
                        + features,
                        mask=active & feature_mask,
                        other=0,
                    )
                else:
                    groups = tl.load(
                        dim_groups_pointer + features,
                        mask=active & feature_mask,
                        other=0,
                    )
                linear = tl.load(
                    linear_pointer + query_head.to(tl.int64) * GMAX + groups,
                    mask=active & feature_mask,
                    other=0.0,
                ).to(tl.float32)
                if WRITE_V:
                    constant = tl.load(
                        constant_pointer
                        + query_head.to(tl.int64) * stride_constant_head,
                        mask=active,
                        other=0.0,
                    ).to(tl.float32)
                    a0 += constant * gradient
                a1 += (linear * query * scale)[:, None] * gradient[None, :]

            if WRITE_V:
                d_v = a0 + tl.sum(key[:, None] * a1, axis=0)
                tl.store(
                    d_v_accumulator_pointer
                    + batch_key_value_head_64 * stride_d_v_batch
                    + local_token.to(tl.int64) * stride_d_v_token
                    + values,
                    d_v,
                    mask=active & value_mask,
                )
            if WRITE_K:
                d_k_partial = tl.sum(a1 * augmented_value[None, :], axis=1)
                tl.store(
                    d_k_partials_pointer
                    + batch_key_value_head_64 * stride_d_k_batch
                    + local_token.to(tl.int64) * stride_d_k_token
                    + value_tile * stride_d_k_value_tile
                    + features,
                    d_k_partial,
                    mask=active & feature_mask,
                )

        if WRITE_V:
            tl.store(
                a0_pointer
                + batch_key_value_head_64 * stride_a0_batch
                + values * stride_a0_channel,
                a0,
                mask=augmented_value_mask,
            )
        tl.store(
            a1_pointer
            + batch_key_value_head_64 * stride_a1_batch
            + features[:, None] * stride_a1_feature
            + values[None, :] * stride_a1_channel,
            a1,
            mask=feature_mask[:, None] & augmented_value_mask[None, :],
        )

    @triton.jit
    def _reverse_a2_all_pair_kernel(
        q_pointer,
        k_pointer,
        v_pointer,
        dim_groups_pointer,
        quadratic_pointer,
        gradient_pointer,
        pair_rows_pointer,
        pair_columns_pointer,
        a2_pointer,
        d_v_pair_partials_pointer,
        d_k_pair_partials_pointer,
        chunk_start,
        chunk_length,
        query_heads,
        key_value_heads,
        head_dimension,
        value_dimension,
        pair_count,
        scale,
        stride_q_batch,
        stride_q_head,
        stride_q_token,
        stride_q_feature,
        stride_k_batch,
        stride_k_head,
        stride_k_token,
        stride_k_feature,
        stride_v_batch,
        stride_v_head,
        stride_v_token,
        stride_v_value,
        stride_gradient_batch,
        stride_gradient_head,
        stride_gradient_token,
        stride_gradient_value,
        stride_a2_batch,
        stride_a2_pair,
        stride_a2_channel,
        stride_d_v_partial_batch,
        stride_d_v_partial_group,
        stride_d_v_partial_token,
        stride_d_k_partial_batch,
        stride_d_k_partial_group,
        stride_d_k_partial_token,
        stride_d_k_partial_value_tile,
        GROUPS_PER_HEAD: tl.constexpr,
        GMAX: tl.constexpr,
        D_PAD: tl.constexpr,
        GQA: tl.constexpr,
        BT: tl.constexpr,
        BP: tl.constexpr,
        BV: tl.constexpr,
        WRITE_K: tl.constexpr,
        WRITE_V: tl.constexpr,
    ):
        local_pair_group = tl.program_id(0)
        value_tile = tl.program_id(1)
        batch_key_value_head = tl.program_id(2)
        batch_key_value_head_64 = batch_key_value_head.to(tl.int64)
        batch = batch_key_value_head // key_value_heads
        key_value_head = batch_key_value_head % key_value_heads
        pairs = local_pair_group * BP + tl.arange(0, BP)
        values = value_tile * BV + tl.arange(0, BV)
        features = tl.arange(0, D_PAD)
        pair_mask = pairs < pair_count
        value_mask = values < value_dimension
        augmented_value_mask = values < value_dimension + 1
        feature_mask = features < head_dimension
        rows = tl.load(pair_rows_pointer + pairs, mask=pair_mask, other=0)
        columns = tl.load(pair_columns_pointer + pairs, mask=pair_mask, other=0)
        a2 = tl.load(
            a2_pointer
            + batch_key_value_head_64 * stride_a2_batch
            + pairs[:, None] * stride_a2_pair
            + values[None, :] * stride_a2_channel,
            mask=pair_mask[:, None] & augmented_value_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        for reverse_step in tl.range(0, BT):
            active = reverse_step < chunk_length
            local_token = chunk_length - 1 - reverse_step
            token = chunk_start + local_token
            key_rows = tl.load(
                k_pointer
                + batch.to(tl.int64) * stride_k_batch
                + key_value_head.to(tl.int64) * stride_k_head
                + token.to(tl.int64) * stride_k_token
                + rows * stride_k_feature,
                mask=active & pair_mask,
                other=0.0,
            ).to(tl.float32)
            key_columns = tl.load(
                k_pointer
                + batch.to(tl.int64) * stride_k_batch
                + key_value_head.to(tl.int64) * stride_k_head
                + token.to(tl.int64) * stride_k_token
                + columns * stride_k_feature,
                mask=active & pair_mask,
                other=0.0,
            ).to(tl.float32)
            loaded_value = tl.load(
                v_pointer
                + batch.to(tl.int64) * stride_v_batch
                + key_value_head.to(tl.int64) * stride_v_head
                + token.to(tl.int64) * stride_v_token
                + values * stride_v_value,
                mask=active & value_mask,
                other=0.0,
            ).to(tl.float32)
            augmented_value = tl.where(
                active & (values == value_dimension), 1.0, loaded_value
            )

            for local_query in range(0, GQA):
                query_head = key_value_head * GQA + local_query
                gradient = tl.load(
                    gradient_pointer
                    + batch.to(tl.int64) * stride_gradient_batch
                    + query_head.to(tl.int64) * stride_gradient_head
                    + token.to(tl.int64) * stride_gradient_token
                    + values * stride_gradient_value,
                    mask=active & augmented_value_mask,
                    other=0.0,
                ).to(tl.float32)
                query_rows = tl.load(
                    q_pointer
                    + batch.to(tl.int64) * stride_q_batch
                    + query_head.to(tl.int64) * stride_q_head
                    + token.to(tl.int64) * stride_q_token
                    + rows * stride_q_feature,
                    mask=active & pair_mask,
                    other=0.0,
                ).to(tl.float32)
                query_columns = tl.load(
                    q_pointer
                    + batch.to(tl.int64) * stride_q_batch
                    + query_head.to(tl.int64) * stride_q_head
                    + token.to(tl.int64) * stride_q_token
                    + columns * stride_q_feature,
                    mask=active & pair_mask,
                    other=0.0,
                ).to(tl.float32)
                if GROUPS_PER_HEAD:
                    row_groups = tl.load(
                        dim_groups_pointer
                        + query_head.to(tl.int64) * head_dimension
                        + rows,
                        mask=active & pair_mask,
                        other=0,
                    )
                    column_groups = tl.load(
                        dim_groups_pointer
                        + query_head.to(tl.int64) * head_dimension
                        + columns,
                        mask=active & pair_mask,
                        other=0,
                    )
                else:
                    row_groups = tl.load(
                        dim_groups_pointer + rows,
                        mask=active & pair_mask,
                        other=0,
                    )
                    column_groups = tl.load(
                        dim_groups_pointer + columns,
                        mask=active & pair_mask,
                        other=0,
                    )
                quadratic = tl.load(
                    quadratic_pointer
                    + query_head.to(tl.int64) * (GMAX * GMAX)
                    + row_groups * GMAX
                    + column_groups,
                    mask=active & pair_mask,
                    other=0.0,
                ).to(tl.float32)
                multiplicity = tl.where(rows == columns, 1.0, 2.0)
                a2 += (
                    multiplicity
                    * quadratic
                    * query_rows
                    * query_columns
                    * scale
                    * scale
                )[:, None] * gradient[None, :]

            if WRITE_V:
                d_v_pair = tl.sum((key_rows * key_columns)[:, None] * a2, axis=0)
                tl.store(
                    d_v_pair_partials_pointer
                    + batch_key_value_head_64 * stride_d_v_partial_batch
                    + local_pair_group * stride_d_v_partial_group
                    + local_token.to(tl.int64) * stride_d_v_partial_token
                    + values,
                    d_v_pair,
                    mask=active & value_mask,
                )
            if WRITE_K:
                pair_gradient = tl.sum(a2 * augmented_value[None, :], axis=1)
                row_match = rows[:, None] == features[None, :]
                column_match = columns[:, None] == features[None, :]
                diagonal = rows == columns
                key_derivative = tl.where(
                    diagonal[:, None] & row_match,
                    2.0 * key_rows[:, None],
                    tl.where(
                        row_match,
                        key_columns[:, None],
                        tl.where(column_match, key_rows[:, None], 0.0),
                    ),
                )
                d_k_pair = tl.sum(pair_gradient[:, None] * key_derivative, axis=0)
                tl.store(
                    d_k_pair_partials_pointer
                    + batch_key_value_head_64 * stride_d_k_partial_batch
                    + local_pair_group * stride_d_k_partial_group
                    + local_token.to(tl.int64) * stride_d_k_partial_token
                    + value_tile * stride_d_k_partial_value_tile
                    + features,
                    d_k_pair,
                    mask=active & feature_mask,
                )

        tl.store(
            a2_pointer
            + batch_key_value_head_64 * stride_a2_batch
            + pairs[:, None] * stride_a2_pair
            + values[None, :] * stride_a2_channel,
            a2,
            mask=pair_mask[:, None] & augmented_value_mask[None, :],
        )

    @triton.jit
    def _reduce_reverse_all_pair_and_write_kernel(
        d_v_accumulator_pointer,
        d_k_partials_pointer,
        d_v_pair_partials_pointer,
        d_k_pair_partials_pointer,
        d_k_pointer,
        d_v_pointer,
        chunk_start,
        chunk_length,
        key_value_heads,
        head_dimension,
        value_dimension,
        stride_d_v_accumulator_batch,
        stride_d_v_accumulator_token,
        stride_d_k_partial_batch,
        stride_d_k_partial_token,
        stride_d_k_partial_value_tile,
        stride_d_v_pair_batch,
        stride_d_v_pair_group,
        stride_d_v_pair_token,
        stride_d_k_pair_batch,
        stride_d_k_pair_group,
        stride_d_k_pair_token,
        stride_d_k_pair_value_tile,
        stride_d_k_batch,
        stride_d_k_head,
        stride_d_k_token,
        stride_d_k_feature,
        stride_d_v_batch,
        stride_d_v_head,
        stride_d_v_token,
        stride_d_v_value,
        D_PAD: tl.constexpr,
        V_PAD: tl.constexpr,
        BT: tl.constexpr,
        VALUE_TILES: tl.constexpr,
        PAIR_GROUPS: tl.constexpr,
        WRITE_K: tl.constexpr,
        WRITE_V: tl.constexpr,
    ):
        # A token/KV-head owner consumes the all-pair producer in canonical
        # ascending packed-group order, then writes dK/dV once.  This replaces
        # the former separate pair reducer and final writer without atomics.
        local_token = tl.program_id(0)
        batch_key_value_head = tl.program_id(1)
        batch_key_value_head_64 = batch_key_value_head.to(tl.int64)
        batch = batch_key_value_head // key_value_heads
        key_value_head = batch_key_value_head % key_value_heads
        active = local_token < chunk_length
        token = chunk_start + local_token

        if WRITE_V:
            values = tl.arange(0, V_PAD)
            value_mask = values < value_dimension
            d_v = tl.load(
                d_v_accumulator_pointer
                + batch_key_value_head_64 * stride_d_v_accumulator_batch
                + local_token.to(tl.int64) * stride_d_v_accumulator_token
                + values,
                mask=active & value_mask,
                other=0.0,
            ).to(tl.float32)
            for local_pair_group in range(0, PAIR_GROUPS):
                d_v += tl.load(
                    d_v_pair_partials_pointer
                    + batch_key_value_head_64 * stride_d_v_pair_batch
                    + local_pair_group * stride_d_v_pair_group
                    + local_token.to(tl.int64) * stride_d_v_pair_token
                    + values,
                    mask=active & value_mask,
                    other=0.0,
                ).to(tl.float32)
            tl.store(
                d_v_pointer
                + batch.to(tl.int64) * stride_d_v_batch
                + key_value_head.to(tl.int64) * stride_d_v_head
                + token.to(tl.int64) * stride_d_v_token
                + values * stride_d_v_value,
                d_v,
                mask=active & value_mask,
            )

        if WRITE_K:
            features = tl.arange(0, D_PAD)
            feature_mask = features < head_dimension
            d_k = tl.zeros((D_PAD,), dtype=tl.float32)
            for value_tile in range(0, VALUE_TILES):
                # Preserve the former fixed reduction tree: first reduce all
                # pair groups for one augmented-value tile, then add that
                # completed tile to the final dK owner.
                d_k_for_value_tile = tl.load(
                    d_k_partials_pointer
                    + batch_key_value_head_64 * stride_d_k_partial_batch
                    + local_token.to(tl.int64) * stride_d_k_partial_token
                    + value_tile * stride_d_k_partial_value_tile
                    + features,
                    mask=active & feature_mask,
                    other=0.0,
                ).to(tl.float32)
                for local_pair_group in range(0, PAIR_GROUPS):
                    d_k_for_value_tile += tl.load(
                        d_k_pair_partials_pointer
                        + batch_key_value_head_64 * stride_d_k_pair_batch
                        + local_pair_group * stride_d_k_pair_group
                        + local_token.to(tl.int64) * stride_d_k_pair_token
                        + value_tile * stride_d_k_pair_value_tile
                        + features,
                        mask=active & feature_mask,
                        other=0.0,
                    ).to(tl.float32)
                d_k += d_k_for_value_tile
            tl.store(
                d_k_pointer
                + batch.to(tl.int64) * stride_d_k_batch
                + key_value_head.to(tl.int64) * stride_d_k_head
                + token.to(tl.int64) * stride_d_k_token
                + features * stride_d_k_feature,
                d_k,
                mask=active & feature_mask,
            )


def _validate_triton_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    grad_augmented: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
) -> None:
    if not all(
        isinstance(tensor, torch.Tensor)
        for tensor in (q, k, v, dim_groups, grad_augmented, constant, linear, quadratic)
    ):
        raise TypeError("grouped causal suffix inputs must be tensors")
    if not triton_is_available():
        raise RuntimeError("grouped causal suffix kernels require Triton")
    if (
        q.ndim != 4
        or k.ndim != 4
        or v.ndim != 4
        or grad_augmented.ndim != 4
        or q.shape[0] != k.shape[0]
        or q.shape[0] != v.shape[0]
        or q.shape[2] != k.shape[2]
        or q.shape[2] != v.shape[2]
    ):
        raise ValueError("grouped causal suffix requires compatible rank-4 Q/K/V/G")
    if k.shape[1] != v.shape[1]:
        raise ValueError("grouped causal suffix requires matching K/V head counts")
    if not all(dimension > 0 for tensor in (q, k, v) for dimension in tensor.shape):
        raise ValueError("grouped causal suffix requires positive Q/K/V dimensions")
    if (
        q.dtype not in {torch.float16, torch.bfloat16}
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise TypeError(
            "grouped causal suffix kernels support matching float16/bfloat16 Q/K/V"
        )
    if grad_augmented.dtype != torch.float32:
        raise TypeError(
            "grouped causal suffix kernels require FP32 augmented gradients"
        )
    if q.shape[-1] != k.shape[-1] or not 1 <= q.shape[-1] <= 64:
        raise ValueError("grouped causal suffix kernels support D through 64")
    if not 1 <= v.shape[-1] <= 64:
        raise ValueError("grouped causal suffix kernels support DV through 64")
    if q.shape[1] % k.shape[1] != 0 or q.shape[1] // k.shape[1] > 16:
        raise ValueError(
            "grouped causal suffix kernels support native GQA ratios through 16"
        )
    if dim_groups.dtype != torch.int32:
        raise TypeError("dim_groups must be int32")
    if any(tensor.dtype != torch.float32 for tensor in (constant, linear, quadratic)):
        raise TypeError("grouped causal suffix coefficients must be FP32")
    gmax = linear.shape[-1] if linear.ndim == 2 else 0
    if (
        constant.shape != (q.shape[1],)
        or linear.shape != (q.shape[1], gmax)
        or quadratic.shape != (q.shape[1], gmax, gmax)
    ):
        raise ValueError(
            "grouped causal suffix requires expanded [Hq]/[Hq,G]/[Hq,G,G] coefficients"
        )
    if dim_groups.ndim == 1:
        valid_groups_shape = dim_groups.shape == (q.shape[-1],)
    elif dim_groups.ndim == 2:
        valid_groups_shape = dim_groups.shape == (q.shape[1], q.shape[-1])
    else:
        valid_groups_shape = False
    if not valid_groups_shape:
        raise ValueError(
            "dim_groups must have canonical shared [D] or per-head [Hq,D] shape"
        )
    if grad_augmented.shape != (q.shape[0], q.shape[1], q.shape[2], v.shape[-1] + 1):
        raise ValueError("G must have exact [B,Hq,N,DV+1] shape")
    if not all(
        tensor.device == q.device
        for tensor in (k, v, dim_groups, grad_augmented, constant, linear, quadratic)
    ):
        raise ValueError("grouped causal suffix tensors must share one device")
    if (
        gmax <= 0
        or bool(torch.any(dim_groups < 0))
        or bool(torch.any(dim_groups >= gmax))
    ):
        raise ValueError("dim_groups must index the expanded coefficient groups")
    if not all(
        tensor.is_cuda
        for tensor in (q, k, v, dim_groups, grad_augmented, constant, linear, quadratic)
    ):
        raise RuntimeError("grouped causal suffix kernels require CUDA tensors")
    if not all(
        tensor.is_contiguous()
        for tensor in (q, k, v, dim_groups, grad_augmented, linear, quadratic)
    ):
        raise ValueError(
            "planned grouped causal suffix requires contiguous Q/K/V/G/groups/B/C"
        )
    if constant.ndim != 1 or constant.stride(0) <= 0:
        raise ValueError(
            "planned grouped causal suffix requires a positive-stride [Hq] A"
        )


def _validate_suffix_plan(
    plan: KernelPlan,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    grad_augmented: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    *,
    requested_gradient_mask: tuple[bool, bool, bool, bool, bool, bool],
) -> tuple[int, int, int]:
    """Fail closed unless the live suffix matches the exact planned geometry."""
    from .grouped_quadratic_causal_backward import (
        _require_live_backward_prefix_plan_identity,
        _runtime_backward_prefix_geometry,
    )

    if not isinstance(plan, KernelPlan):
        raise TypeError("planned grouped causal suffix requires a KernelPlan")
    if plan.geometry.workspace_budget_bytes is None:
        raise ValueError("suffix kernel plan must carry a resolved workspace budget")
    current_geometry = _runtime_backward_prefix_geometry(
        q,
        k,
        v,
        dim_groups,
        grad_augmented,
        constant,
        linear,
        quadratic,
        needs_input_grad=requested_gradient_mask,
        workspace_budget_bytes=plan.geometry.workspace_budget_bytes,
    )
    if current_geometry.to_dict() != plan.geometry.to_dict():
        raise ValueError(
            "suffix kernel plan is stale for the live CUDA/tensor/toolchain geometry"
        )
    live_plan = _require_live_backward_prefix_plan_identity(plan, current_geometry)
    free_bytes, _ = torch.cuda.mem_get_info(q.device)
    if min(512 * 2**20, free_bytes // 20) < live_plan.geometry.workspace_budget_bytes:
        raise ValueError("live free memory cannot honor the suffix kernel plan")
    config = planned_grouped_causal_scan_config(live_plan)
    suffix_workspace_allocation_names(live_plan)
    return (
        config.token_block,
        config.pair_block,
        config.value_block,
    )


def grouped_causal_triton_reverse_dk_dv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    grad_augmented: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    *,
    scale: float,
    need_k: bool,
    need_v: bool,
    requested_gradient_mask: tuple[bool, bool, bool, bool, bool, bool] | None = None,
    plan: KernelPlan | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return FP32 dK/dV through the exact planned reverse scan workspace."""
    if not need_k and not need_v:
        return None, None
    _validate_triton_inputs(
        q, k, v, dim_groups, grad_augmented, constant, linear, quadratic
    )
    assert triton is not None
    requested_mask = (
        (False, need_k, need_v, False, False, False)
        if requested_gradient_mask is None
        else tuple(requested_gradient_mask)
    )
    if len(requested_mask) != 6 or not all(
        isinstance(value, bool) for value in requested_mask
    ):
        raise TypeError("requested_gradient_mask must contain six Q/K/V/A/B/C booleans")
    if requested_mask[1:3] != (need_k, need_v):
        raise ValueError("requested_gradient_mask must agree with suffix outputs")
    if plan is None:
        # Compatibility callers still receive the same strict internal plan;
        # public admission remains fail-closed until evidence is reviewed.
        from .grouped_quadratic_causal_backward import build_backward_kernel_plan

        plan = build_backward_kernel_plan(
            q,
            k,
            v,
            dim_groups,
            grad_augmented,
            constant,
            linear,
            quadratic,
            needs_input_grad=requested_mask,
        )
    token_block, pair_block, value_block = _validate_suffix_plan(
        plan,
        q,
        k,
        v,
        dim_groups,
        grad_augmented,
        constant,
        linear,
        quadratic,
        requested_gradient_mask=requested_mask,
    )
    batch_size, query_heads, token_count, head_dimension = q.shape
    key_value_heads = k.shape[1]
    value_dimension = v.shape[-1]
    augmented_dimension = value_dimension + 1
    gmax = linear.shape[-1]
    pair_count = plan.geometry.pair_count
    value_tiles = triton.cdiv(augmented_dimension, value_block)
    pair_groups = triton.cdiv(pair_count, pair_block)
    batch_key_value_heads = batch_size * key_value_heads
    groups_per_head = dim_groups.ndim == 2
    gqa = query_heads // key_value_heads
    d_pad = triton.next_power_of_2(head_dimension)
    workspace = allocate_planned_suffix_workspace(plan, device=q.device)
    a0, a1, a2 = _suffix_state_views(plan, workspace)
    placeholder = grad_augmented
    pair_rows = workspace["suffix_pair_rows"]
    pair_columns = workspace["suffix_pair_columns"]
    d_v_accumulator = (
        _flatten_suffix_kv_workspace(
            workspace["suffix_dv_macro"], name="suffix_dv_macro"
        )
        if need_v
        else placeholder
    )
    d_k_partials = (
        _flatten_suffix_kv_workspace(
            workspace["suffix_dk_value_partial"], name="suffix_dk_value_partial"
        )
        if need_k
        else placeholder
    )
    d_v_pair_partials = (
        _flatten_suffix_kv_workspace(
            workspace["suffix_dv_pair_partial"], name="suffix_dv_pair_partial"
        )
        if need_v
        else placeholder
    )
    d_k_pair_partials = (
        _flatten_suffix_kv_workspace(
            workspace["suffix_dk_pair_partial"], name="suffix_dk_pair_partial"
        )
        if need_k
        else placeholder
    )
    d_k = workspace.get("dk", k)
    d_v = workspace.get("dv", v)

    for chunk_start in range(
        ((token_count - 1) // token_block) * token_block,
        -1,
        -token_block,
    ):
        chunk_length = min(token_block, token_count - chunk_start)
        _reverse_a0_a1_token_block_kernel[(value_tiles, batch_key_value_heads)](
            q,
            k,
            v,
            dim_groups,
            constant,
            linear,
            grad_augmented,
            a0,
            a1,
            d_v_accumulator,
            d_k_partials,
            chunk_start,
            chunk_length,
            query_heads,
            key_value_heads,
            head_dimension,
            value_dimension,
            scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            grad_augmented.stride(0),
            grad_augmented.stride(1),
            grad_augmented.stride(2),
            grad_augmented.stride(3),
            constant.stride(0),
            a0.stride(0),
            a0.stride(1),
            a1.stride(0),
            a1.stride(1),
            a1.stride(2),
            d_v_accumulator.stride(0),
            d_v_accumulator.stride(1),
            d_k_partials.stride(0),
            d_k_partials.stride(1),
            d_k_partials.stride(2),
            GROUPS_PER_HEAD=groups_per_head,
            GMAX=gmax,
            D_PAD=d_pad,
            GQA=gqa,
            BT=token_block,
            BV=value_block,
            WRITE_K=need_k,
            WRITE_V=need_v,
            num_warps=4,
        )
        _reverse_a2_all_pair_kernel[(pair_groups, value_tiles, batch_key_value_heads)](
            q,
            k,
            v,
            dim_groups,
            quadratic,
            grad_augmented,
            pair_rows,
            pair_columns,
            a2,
            d_v_pair_partials,
            d_k_pair_partials,
            chunk_start,
            chunk_length,
            query_heads,
            key_value_heads,
            head_dimension,
            value_dimension,
            pair_count,
            scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            grad_augmented.stride(0),
            grad_augmented.stride(1),
            grad_augmented.stride(2),
            grad_augmented.stride(3),
            a2.stride(0),
            a2.stride(1),
            a2.stride(2),
            d_v_pair_partials.stride(0),
            d_v_pair_partials.stride(1),
            d_v_pair_partials.stride(2),
            d_k_pair_partials.stride(0),
            d_k_pair_partials.stride(1),
            d_k_pair_partials.stride(2),
            d_k_pair_partials.stride(3),
            GROUPS_PER_HEAD=groups_per_head,
            GMAX=gmax,
            D_PAD=d_pad,
            GQA=gqa,
            BT=token_block,
            BP=pair_block,
            BV=value_block,
            WRITE_K=need_k,
            WRITE_V=need_v,
            num_warps=4,
        )
        _reduce_reverse_all_pair_and_write_kernel[(token_block, batch_key_value_heads)](
            d_v_accumulator,
            d_k_partials,
            d_v_pair_partials,
            d_k_pair_partials,
            d_k,
            d_v,
            chunk_start,
            chunk_length,
            key_value_heads,
            head_dimension,
            value_dimension,
            d_v_accumulator.stride(0),
            d_v_accumulator.stride(1),
            d_k_partials.stride(0),
            d_k_partials.stride(1),
            d_k_partials.stride(2),
            d_v_pair_partials.stride(0),
            d_v_pair_partials.stride(1),
            d_v_pair_partials.stride(2),
            d_k_pair_partials.stride(0),
            d_k_pair_partials.stride(1),
            d_k_pair_partials.stride(2),
            d_k_pair_partials.stride(3),
            d_k.stride(0),
            d_k.stride(1),
            d_k.stride(2),
            d_k.stride(3),
            d_v.stride(0),
            d_v.stride(1),
            d_v.stride(2),
            d_v.stride(3),
            D_PAD=d_pad,
            V_PAD=triton.next_power_of_2(value_dimension),
            VALUE_TILES=value_tiles,
            BT=token_block,
            PAIR_GROUPS=pair_groups,
            WRITE_K=need_k,
            WRITE_V=need_v,
            num_warps=4,
        )
    _capture_suffix_production_witness(a2, plan=plan)
    return (d_k if need_k else None, d_v if need_v else None)


__all__ = (
    "grouped_causal_triton_reverse_dk_dv",
    "suffix_production_witness_capture_names",
    "triton_is_available",
)
