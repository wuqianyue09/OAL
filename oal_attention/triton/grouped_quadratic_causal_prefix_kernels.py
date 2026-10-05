"""CUDA kernels for the KV-owned grouped causal forward-prefix VJP scan.

This module owns only the bounded Triton schedule for ``dQ/dA/dB/dC``.  H0/H1
advance in token blocks, H2 advances over the complete packed-pair grid, and
every query-head fan-in uses an explicit fixed-order partial/reduction buffer.
The Python launcher and its CPU fallback remain in
``grouped_quadratic_causal_backward``.
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
    """Return whether the dedicated grouped causal prefix kernels are usable."""
    return triton is not None and tl is not None


_PREFIX_PRODUCTION_WITNESS_CAPTURE_NAMES: Final[tuple[str, ...]] = (
    "prefix.h_inclusive",
)


def prefix_production_witness_capture_names() -> tuple[str, ...]:
    """Return the scalar probe emitted from the actual final H-prefix state."""
    return _PREFIX_PRODUCTION_WITNESS_CAPTURE_NAMES


def _capture_prefix_production_witness(
    h2: torch.Tensor,
    *,
    plan: KernelPlan,
) -> None:
    session = active_production_witness_session()
    if session is None or not session.expects("prefix.h_inclusive"):
        return
    probe = session.request.probe_for("prefix")
    geometry = plan.geometry
    if (
        probe.batch_index >= geometry.B
        or probe.key_value_head >= geometry.Hkv
        or probe.query_head >= geometry.Hq
        or probe.token_index != geometry.N - 1
        or probe.pair_index >= geometry.pair_count
        or probe.channel_index >= geometry.augmented_value_dimension
        or probe.query_head // (geometry.Hq // geometry.Hkv) != probe.key_value_head
    ):
        raise ValueError(
            "prefix production witness probe is not a final GQA coordinate"
        )
    flat_key_value_head = probe.batch_index * geometry.Hkv + probe.key_value_head
    session.capture_device_tensor(
        "prefix.h_inclusive",
        h2[flat_key_value_head, probe.pair_index, probe.channel_index].reshape(1),
    )


def _torch_dtype(dtype_name: str) -> torch.dtype:
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"unsupported planned tensor dtype: {dtype_name}")
    return dtype


def prefix_workspace_allocation_names(plan: KernelPlan) -> tuple[str, ...]:
    """Return physical records; pair row/column aliases are runtime-only views."""
    if not isinstance(plan, KernelPlan):
        raise TypeError("planned grouped causal prefix requires a KernelPlan")
    needs_q, _, _, needs_a, needs_b, needs_c = plan.geometry.requested_gradient_mask
    if not (needs_q or needs_a or needs_b or needs_c):
        return ()
    names = ["prefix_state"]
    if needs_q or needs_c:
        names.extend(
            (
                "prefix_pair_metadata",
                "prefix_h2_dot_partial",
            )
        )
    if needs_a:
        names.extend(("prefix_h0_dot_partial", "d_a_macro", "d_constant"))
    if needs_q or needs_b:
        names.append("prefix_h1_dot_partial")
    if needs_q:
        names.extend(("prefix_dq_macro", "dq"))
    if needs_c:
        names.extend(("d_c_macro", "d_quadratic"))
    if needs_b:
        names.extend(("d_b_macro", "d_linear"))
    for name in names:
        allocation = plan.workspace.allocation(name)
        if allocation.alias_of is not None:
            raise ValueError(f"prefix workspace allocation {name} cannot be an alias")
    return tuple(names)


def allocate_planned_prefix_workspace(
    plan: KernelPlan,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Allocate physical records and expose pair-row/column runtime alias views."""
    buffers: dict[str, torch.Tensor] = {}
    names = prefix_workspace_allocation_names(plan)
    for name in names:
        if name == "prefix_pair_metadata":
            continue
        allocation = plan.workspace.allocation(name)
        kwargs = {"device": device, "dtype": _torch_dtype(allocation.dtype)}
        buffers[name] = (
            torch.zeros(allocation.shape, **kwargs)
            if name == "prefix_state"
            else torch.empty(allocation.shape, **kwargs)
        )
    if "prefix_pair_metadata" in names:
        allocation = plan.workspace.allocation("prefix_pair_metadata")
        pair_metadata = torch.tril_indices(
            plan.geometry.D,
            plan.geometry.D,
            device=device,
            dtype=torch.int32,
        )
        if tuple(
            pair_metadata.shape
        ) != allocation.shape or pair_metadata.dtype != _torch_dtype(allocation.dtype):
            raise AssertionError("canonical prefix pair metadata does not match plan")
        for name in ("prefix_pair_rows", "prefix_pair_columns"):
            alias = plan.workspace.allocation(name)
            if alias.alias_of != "prefix_pair_metadata":
                raise ValueError(f"{name} must alias prefix_pair_metadata")
        buffers["prefix_pair_metadata"] = pair_metadata
        # These are views of the exact torch.tril_indices allocation: no copy,
        # contiguous conversion, or second metadata allocation is permitted.
        buffers["prefix_pair_rows"] = pair_metadata[0]
        buffers["prefix_pair_columns"] = pair_metadata[1]
    return buffers


def _prefix_state_views(
    plan: KernelPlan,
    workspace: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create stride-aware H0/H1/H2 views of the one planned FP32 state set."""
    state = workspace["prefix_state"]
    geometry = plan.geometry
    expected_shape = (
        geometry.B,
        geometry.Hkv,
        geometry.augmented_value_dimension,
        1 + geometry.D + geometry.pair_count,
    )
    if state.dtype != torch.float32 or tuple(state.shape) != expected_shape:
        raise ValueError("prefix_state does not match the exact FP32 plan record")
    for name in ("prefix_h0_view", "prefix_h1_view", "prefix_h2_view"):
        if plan.workspace.allocation(name).alias_of != "prefix_state":
            raise ValueError(f"{name} must remain a prefix_state alias")
    block_stride = state.stride(1)
    channel_stride = state.stride(2)
    feature_stride = state.stride(3)
    batch_key_value_heads = geometry.B * geometry.Hkv
    h0 = state.as_strided(
        (batch_key_value_heads, geometry.augmented_value_dimension),
        (block_stride, channel_stride),
    )
    h1 = state.as_strided(
        (batch_key_value_heads, geometry.D, geometry.augmented_value_dimension),
        (block_stride, feature_stride, channel_stride),
        storage_offset=1,
    )
    h2 = state.as_strided(
        (
            batch_key_value_heads,
            geometry.pair_count,
            geometry.augmented_value_dimension,
        ),
        (block_stride, feature_stride, channel_stride),
        storage_offset=1 + geometry.D,
    )
    return h0, h1, h2


if triton_is_available():

    @triton.jit
    def _prefix_h0_h1_token_block_kernel(
        k_pointer,
        v_pointer,
        gradient_pointer,
        h0_pointer,
        h1_pointer,
        h0_dot_partials_pointer,
        h1_dot_partials_pointer,
        chunk_start,
        chunk_length,
        query_heads,
        key_value_heads,
        head_dimension,
        value_dimension,
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
        stride_h0_batch,
        stride_h0_channel,
        stride_h1_batch,
        stride_h1_feature,
        stride_h1_channel,
        stride_h0_partial_batch,
        stride_h0_partial_token,
        stride_h1_partial_batch,
        stride_h1_partial_token,
        stride_h1_partial_value_tile,
        D_PAD: tl.constexpr,
        GQA: tl.constexpr,
        BT: tl.constexpr,
        BV: tl.constexpr,
        WRITE_CONSTANT: tl.constexpr,
        WRITE_H1_DOT: tl.constexpr,
    ):
        value_tile = tl.program_id(0)
        batch_key_value_head = tl.program_id(1)
        batch_key_value_head_64 = batch_key_value_head.to(tl.int64)
        batch = batch_key_value_head // key_value_heads
        key_value_head = batch_key_value_head % key_value_heads
        values = value_tile * BV + tl.arange(0, BV)
        features = tl.arange(0, D_PAD)
        feature_mask = features < head_dimension
        augmented_value_mask = values < value_dimension + 1
        value_mask = values < value_dimension
        if WRITE_CONSTANT:
            h0 = tl.load(
                h0_pointer
                + batch_key_value_head_64 * stride_h0_batch
                + values * stride_h0_channel,
                mask=augmented_value_mask,
                other=0.0,
            ).to(tl.float32)
        else:
            h0 = tl.zeros((BV,), dtype=tl.float32)
        h1 = tl.load(
            h1_pointer
            + batch_key_value_head_64 * stride_h1_batch
            + features[:, None] * stride_h1_feature
            + values[None, :] * stride_h1_channel,
            mask=feature_mask[:, None] & augmented_value_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        for local_token in tl.range(0, BT):
            active = local_token < chunk_length
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
            if WRITE_CONSTANT:
                h0 += augmented_value
            h1 += key[:, None] * augmented_value[None, :]

            for local_query in range(0, GQA):
                query_head = key_value_head * GQA + local_query
                batch_query_head = batch * query_heads + query_head
                batch_query_head_64 = batch_query_head.to(tl.int64)
                gradient = tl.load(
                    gradient_pointer
                    + batch.to(tl.int64) * stride_gradient_batch
                    + query_head.to(tl.int64) * stride_gradient_head
                    + token.to(tl.int64) * stride_gradient_token
                    + values * stride_gradient_value,
                    mask=active & augmented_value_mask,
                    other=0.0,
                ).to(tl.float32)
                if WRITE_CONSTANT:
                    tl.store(
                        h0_dot_partials_pointer
                        + batch_query_head_64 * stride_h0_partial_batch
                        + local_token * stride_h0_partial_token
                        + value_tile,
                        tl.sum(h0 * gradient, axis=0),
                        mask=active,
                    )
                if WRITE_H1_DOT:
                    tl.store(
                        h1_dot_partials_pointer
                        + batch_query_head_64 * stride_h1_partial_batch
                        + local_token * stride_h1_partial_token
                        + value_tile * stride_h1_partial_value_tile
                        + features,
                        tl.sum(h1 * gradient[None, :], axis=1),
                        mask=active & feature_mask,
                    )

        if WRITE_CONSTANT:
            tl.store(
                h0_pointer
                + batch_key_value_head_64 * stride_h0_batch
                + values * stride_h0_channel,
                h0,
                mask=augmented_value_mask,
            )
        tl.store(
            h1_pointer
            + batch_key_value_head_64 * stride_h1_batch
            + features[:, None] * stride_h1_feature
            + values[None, :] * stride_h1_channel,
            h1,
            mask=feature_mask[:, None] & augmented_value_mask[None, :],
        )

    @triton.jit
    def _reduce_prefix_h0_h1_kernel(
        q_pointer,
        dim_groups_pointer,
        linear_pointer,
        h0_dot_partials_pointer,
        h1_dot_partials_pointer,
        d_q_block_pointer,
        d_a_token_pointer,
        d_b_token_pointer,
        chunk_start,
        chunk_length,
        query_heads,
        head_dimension,
        scale,
        stride_q_batch,
        stride_q_head,
        stride_q_token,
        stride_q_feature,
        stride_h0_partial_batch,
        stride_h0_partial_token,
        stride_h1_partial_batch,
        stride_h1_partial_token,
        stride_h1_partial_value_tile,
        stride_d_q_batch,
        stride_d_q_token,
        stride_d_a_batch,
        stride_d_a_token,
        stride_d_b_batch,
        stride_d_b_token,
        GROUPS_PER_HEAD: tl.constexpr,
        GMAX: tl.constexpr,
        D_PAD: tl.constexpr,
        VALUE_TILES: tl.constexpr,
        BT: tl.constexpr,
        WRITE_Q: tl.constexpr,
        WRITE_CONSTANT: tl.constexpr,
        WRITE_LINEAR: tl.constexpr,
    ):
        local_token = tl.program_id(0)
        batch_query_head = tl.program_id(1)
        batch_query_head_64 = batch_query_head.to(tl.int64)
        batch = batch_query_head // query_heads
        query_head = batch_query_head % query_heads
        active = local_token < chunk_length
        features = tl.arange(0, D_PAD)
        feature_mask = features < head_dimension
        h1_dot = tl.zeros((D_PAD,), dtype=tl.float32)
        h0_dot = 0.0
        if WRITE_Q or WRITE_LINEAR:
            for value_tile in range(0, VALUE_TILES):
                h1_dot += tl.load(
                    h1_dot_partials_pointer
                    + batch_query_head_64 * stride_h1_partial_batch
                    + local_token * stride_h1_partial_token
                    + value_tile * stride_h1_partial_value_tile
                    + features,
                    mask=active & feature_mask,
                    other=0.0,
                ).to(tl.float32)
        if WRITE_CONSTANT:
            for value_tile in range(0, VALUE_TILES):
                h0_dot += tl.load(
                    h0_dot_partials_pointer
                    + batch_query_head_64 * stride_h0_partial_batch
                    + local_token * stride_h0_partial_token
                    + value_tile,
                    mask=active,
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
        if WRITE_Q:
            linear = tl.load(
                linear_pointer + query_head.to(tl.int64) * GMAX + groups,
                mask=active & feature_mask,
                other=0.0,
            ).to(tl.float32)
            tl.store(
                d_q_block_pointer
                + batch_query_head_64 * stride_d_q_batch
                + local_token * stride_d_q_token
                + features,
                scale * linear * h1_dot,
                mask=active & feature_mask,
            )
        if WRITE_CONSTANT:
            token = chunk_start + local_token
            tl.store(
                d_a_token_pointer
                + batch_query_head_64 * stride_d_a_batch
                + local_token * stride_d_a_token,
                h0_dot,
                mask=active,
            )
        if WRITE_LINEAR:
            token = chunk_start + local_token
            query = tl.load(
                q_pointer
                + batch.to(tl.int64) * stride_q_batch
                + query_head.to(tl.int64) * stride_q_head
                + token.to(tl.int64) * stride_q_token
                + features * stride_q_feature,
                mask=active & feature_mask,
                other=0.0,
            ).to(tl.float32)
            group_offsets = tl.arange(0, GMAX)
            d_b = tl.zeros((GMAX,), dtype=tl.float32)
            for group in range(0, GMAX):
                d_b += tl.where(
                    group_offsets == group,
                    tl.sum(
                        tl.where(groups == group, query * scale * h1_dot, 0.0),
                        axis=0,
                    ),
                    0.0,
                )
            tl.store(
                d_b_token_pointer
                + batch_query_head_64 * stride_d_b_batch
                + local_token * stride_d_b_token
                + group_offsets,
                d_b,
                mask=active,
            )

    @triton.jit
    def _prefix_h2_all_pair_kernel(
        k_pointer,
        v_pointer,
        gradient_pointer,
        pair_rows_pointer,
        pair_columns_pointer,
        h2_pointer,
        h2_dot_partials_pointer,
        chunk_start,
        chunk_length,
        query_heads,
        key_value_heads,
        value_dimension,
        pair_count,
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
        stride_h2_batch,
        stride_h2_pair,
        stride_h2_channel,
        stride_h2_partial_batch,
        stride_h2_partial_group,
        stride_h2_partial_token,
        stride_h2_partial_value_tile,
        GQA: tl.constexpr,
        BT: tl.constexpr,
        BP: tl.constexpr,
        BV: tl.constexpr,
    ):
        local_pair_group = tl.program_id(0)
        value_tile = tl.program_id(1)
        batch_key_value_head = tl.program_id(2)
        batch_key_value_head_64 = batch_key_value_head.to(tl.int64)
        batch = batch_key_value_head // key_value_heads
        key_value_head = batch_key_value_head % key_value_heads
        pairs = local_pair_group * BP + tl.arange(0, BP)
        values = value_tile * BV + tl.arange(0, BV)
        pair_mask = pairs < pair_count
        augmented_value_mask = values < value_dimension + 1
        value_mask = values < value_dimension
        rows = tl.load(pair_rows_pointer + pairs, mask=pair_mask, other=0)
        columns = tl.load(pair_columns_pointer + pairs, mask=pair_mask, other=0)
        h2 = tl.load(
            h2_pointer
            + batch_key_value_head_64 * stride_h2_batch
            + pairs[:, None] * stride_h2_pair
            + values[None, :] * stride_h2_channel,
            mask=pair_mask[:, None] & augmented_value_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        for local_token in tl.range(0, BT):
            active = local_token < chunk_length
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
            h2 += (key_rows * key_columns)[:, None] * augmented_value[None, :]
            for local_query in range(0, GQA):
                query_head = key_value_head * GQA + local_query
                batch_query_head = batch * query_heads + query_head
                tl.store(
                    h2_dot_partials_pointer
                    + batch_query_head.to(tl.int64) * stride_h2_partial_batch
                    + local_pair_group * stride_h2_partial_group
                    + local_token * stride_h2_partial_token
                    + value_tile * stride_h2_partial_value_tile
                    + tl.arange(0, BP),
                    tl.sum(
                        h2
                        * tl.load(
                            gradient_pointer
                            + batch.to(tl.int64) * stride_gradient_batch
                            + query_head.to(tl.int64) * stride_gradient_head
                            + token.to(tl.int64) * stride_gradient_token
                            + values * stride_gradient_value,
                            mask=active & augmented_value_mask,
                            other=0.0,
                        ).to(tl.float32)[None, :],
                        axis=1,
                    ),
                    mask=active & pair_mask,
                )
        tl.store(
            h2_pointer
            + batch_key_value_head_64 * stride_h2_batch
            + pairs[:, None] * stride_h2_pair
            + values[None, :] * stride_h2_channel,
            h2,
            mask=pair_mask[:, None] & augmented_value_mask[None, :],
        )

    @triton.jit
    def _reduce_prefix_all_pair_and_accumulate_kernel(
        q_pointer,
        dim_groups_pointer,
        quadratic_pointer,
        pair_rows_pointer,
        pair_columns_pointer,
        h2_dot_partials_pointer,
        d_q_block_pointer,
        d_c_macro_pointer,
        chunk_start,
        chunk_length,
        query_heads,
        head_dimension,
        pair_count,
        scale,
        stride_q_batch,
        stride_q_head,
        stride_q_token,
        stride_q_feature,
        stride_h2_partial_batch,
        stride_h2_partial_group,
        stride_h2_partial_token,
        stride_h2_partial_value_tile,
        stride_d_q_batch,
        stride_d_q_token,
        stride_d_c_macro_batch,
        stride_d_c_macro_token,
        GROUPS_PER_HEAD: tl.constexpr,
        GMAX: tl.constexpr,
        D_PAD: tl.constexpr,
        VALUE_TILES: tl.constexpr,
        BT: tl.constexpr,
        BP: tl.constexpr,
        PAIR_GROUPS: tl.constexpr,
        WRITE_Q: tl.constexpr,
        WRITE_COEFFICIENTS: tl.constexpr,
    ):
        # One owner visits all canonical packed-pair groups in ascending
        # order.  This fuses the former per-group reducer and macro
        # accumulation kernel without changing either H2 physics or the
        # coefficient carry order.
        local_token = tl.program_id(0)
        batch_query_head = tl.program_id(1)
        batch_query_head_64 = batch_query_head.to(tl.int64)
        batch = batch_query_head // query_heads
        query_head = batch_query_head % query_heads
        active = local_token < chunk_length
        token = chunk_start + local_token
        if WRITE_Q:
            features = tl.arange(0, D_PAD)
            feature_mask = features < head_dimension
            d_q = tl.load(
                d_q_block_pointer
                + batch_query_head_64 * stride_d_q_batch
                + local_token * stride_d_q_token
                + features,
                mask=active & feature_mask,
                other=0.0,
            ).to(tl.float32)
        if WRITE_COEFFICIENTS:
            group_pairs = tl.arange(0, GMAX * GMAX)
            group_rows = group_pairs // GMAX
            group_columns = group_pairs % GMAX
            d_c = tl.zeros((GMAX * GMAX,), dtype=tl.float32)

        for local_pair_group in range(0, PAIR_GROUPS):
            pairs = local_pair_group * BP + tl.arange(0, BP)
            pair_mask = pairs < pair_count
            rows = tl.load(pair_rows_pointer + pairs, mask=pair_mask, other=0)
            columns = tl.load(pair_columns_pointer + pairs, mask=pair_mask, other=0)
            h2_dot = tl.zeros((BP,), dtype=tl.float32)
            for value_tile in range(0, VALUE_TILES):
                h2_dot += tl.load(
                    h2_dot_partials_pointer
                    + batch_query_head_64 * stride_h2_partial_batch
                    + local_pair_group * stride_h2_partial_group
                    + local_token * stride_h2_partial_token
                    + value_tile * stride_h2_partial_value_tile
                    + tl.arange(0, BP),
                    mask=active & pair_mask,
                    other=0.0,
                ).to(tl.float32)
            query_rows = (
                tl.load(
                    q_pointer
                    + batch.to(tl.int64) * stride_q_batch
                    + query_head.to(tl.int64) * stride_q_head
                    + token.to(tl.int64) * stride_q_token
                    + rows * stride_q_feature,
                    mask=active & pair_mask,
                    other=0.0,
                ).to(tl.float32)
                * scale
            )
            query_columns = (
                tl.load(
                    q_pointer
                    + batch.to(tl.int64) * stride_q_batch
                    + query_head.to(tl.int64) * stride_q_head
                    + token.to(tl.int64) * stride_q_token
                    + columns * stride_q_feature,
                    mask=active & pair_mask,
                    other=0.0,
                ).to(tl.float32)
                * scale
            )
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
                    dim_groups_pointer + rows, mask=active & pair_mask, other=0
                )
                column_groups = tl.load(
                    dim_groups_pointer + columns, mask=active & pair_mask, other=0
                )
            if WRITE_Q:
                quadratic = tl.load(
                    quadratic_pointer
                    + query_head.to(tl.int64) * (GMAX * GMAX)
                    + row_groups * GMAX
                    + column_groups,
                    mask=active & pair_mask,
                    other=0.0,
                ).to(tl.float32)
                d_q += (
                    tl.sum(
                        tl.where(
                            rows[:, None] == features[None, :],
                            2.0
                            * quadratic[:, None]
                            * query_columns[:, None]
                            * h2_dot[:, None],
                            0.0,
                        )
                        + tl.where(
                            (columns[:, None] == features[None, :])
                            & (rows[:, None] != columns[:, None]),
                            2.0
                            * quadratic[:, None]
                            * query_rows[:, None]
                            * h2_dot[:, None],
                            0.0,
                        ),
                        axis=0,
                    )
                    * scale
                )
            if WRITE_COEFFICIENTS:
                multiplicity = tl.where(rows == columns, 1.0, 2.0)
                contribution = multiplicity * query_rows * query_columns * h2_dot
                d_c += tl.sum(
                    tl.where(
                        (row_groups[:, None] == group_rows[None, :])
                        & (column_groups[:, None] == group_columns[None, :]),
                        contribution[:, None],
                        0.0,
                    ),
                    axis=0,
                )
        if WRITE_Q:
            tl.store(
                d_q_block_pointer
                + batch_query_head_64 * stride_d_q_batch
                + local_token * stride_d_q_token
                + features,
                d_q,
                mask=active & feature_mask,
            )
        if WRITE_COEFFICIENTS:
            tl.store(
                d_c_macro_pointer
                + batch_query_head_64 * stride_d_c_macro_batch
                + local_token * stride_d_c_macro_token
                + group_pairs,
                d_c,
                mask=active,
            )

    @triton.jit
    def _write_prefix_gradients_kernel(
        d_q_block_pointer,
        d_q_pointer,
        chunk_start,
        chunk_length,
        query_heads,
        head_dimension,
        stride_d_q_block_batch,
        stride_d_q_block_token,
        stride_d_q_batch,
        stride_d_q_head,
        stride_d_q_token,
        stride_d_q_feature,
        D_PAD: tl.constexpr,
        BT: tl.constexpr,
    ):
        local_token = tl.program_id(0)
        batch_query_head = tl.program_id(1)
        batch_query_head_64 = batch_query_head.to(tl.int64)
        batch = batch_query_head // query_heads
        query_head = batch_query_head % query_heads
        active = local_token < chunk_length
        features = tl.arange(0, D_PAD)
        feature_mask = features < head_dimension
        d_q = tl.load(
            d_q_block_pointer
            + batch_query_head_64 * stride_d_q_block_batch
            + local_token * stride_d_q_block_token
            + features,
            mask=active & feature_mask,
            other=0.0,
        )
        tl.store(
            d_q_pointer
            + batch.to(tl.int64) * stride_d_q_batch
            + query_head.to(tl.int64) * stride_d_q_head
            + (chunk_start + local_token).to(tl.int64) * stride_d_q_token
            + features * stride_d_q_feature,
            d_q,
            mask=active & feature_mask,
        )

    @triton.jit
    def _coefficient_carry_kernel(
        d_a_macro_pointer,
        d_b_macro_pointer,
        d_c_macro_pointer,
        d_constant_pointer,
        d_linear_pointer,
        d_quadratic_pointer,
        chunk_length,
        query_heads,
        stride_d_a_macro_batch,
        stride_d_a_macro_token,
        stride_d_b_macro_batch,
        stride_d_b_macro_token,
        stride_d_c_macro_batch,
        stride_d_c_macro_token,
        GMAX: tl.constexpr,
        BATCH_SIZE: tl.constexpr,
        BT: tl.constexpr,
        FIRST_CHUNK: tl.constexpr,
        WRITE_CONSTANT: tl.constexpr,
        WRITE_LINEAR: tl.constexpr,
        WRITE_QUADRATIC: tl.constexpr,
    ):
        """Persist one FP32 coefficient carry per Hq in canonical order.

        Macro launches are serialized because H0/H1/H2 are causal.  This
        single-owner program then consumes only the macro-local records in
        batch-major, token-major, group-major, and row/column-major order.
        No atomics or per-macro history participate in the final reduction.
        """
        query_head = tl.program_id(0)
        groups = tl.arange(0, GMAX)
        group_pairs = tl.arange(0, GMAX * GMAX)
        if WRITE_CONSTANT:
            if FIRST_CHUNK:
                d_a = 0.0
            else:
                d_a = tl.load(d_constant_pointer + query_head).to(tl.float32)
        if WRITE_LINEAR:
            if FIRST_CHUNK:
                d_b = tl.zeros((GMAX,), dtype=tl.float32)
            else:
                d_b = tl.load(
                    d_linear_pointer + query_head * GMAX + groups,
                ).to(tl.float32)
        if WRITE_QUADRATIC:
            if FIRST_CHUNK:
                d_c = tl.zeros((GMAX * GMAX,), dtype=tl.float32)
            else:
                d_c = tl.load(
                    d_quadratic_pointer + query_head * (GMAX * GMAX) + group_pairs,
                ).to(tl.float32)
        for batch in range(0, BATCH_SIZE):
            batch_query_head = batch * query_heads + query_head
            for local_token in range(0, BT):
                active = local_token < chunk_length
                if WRITE_CONSTANT:
                    d_a += tl.load(
                        d_a_macro_pointer
                        + batch_query_head * stride_d_a_macro_batch
                        + local_token * stride_d_a_macro_token,
                        mask=active,
                        other=0.0,
                    ).to(tl.float32)
                # The per-group and row/column order is explicit even though
                # each lane owns one final coefficient element.
                if WRITE_LINEAR:
                    for group in range(0, GMAX):
                        d_b += tl.where(
                            groups == group,
                            tl.load(
                                d_b_macro_pointer
                                + batch_query_head * stride_d_b_macro_batch
                                + local_token * stride_d_b_macro_token
                                + groups,
                                mask=active,
                                other=0.0,
                            ).to(tl.float32),
                            0.0,
                        )
                if WRITE_QUADRATIC:
                    for row_column in range(0, GMAX * GMAX):
                        d_c += tl.where(
                            group_pairs == row_column,
                            tl.load(
                                d_c_macro_pointer
                                + batch_query_head * stride_d_c_macro_batch
                                + local_token * stride_d_c_macro_token
                                + group_pairs,
                                mask=active,
                                other=0.0,
                            ).to(tl.float32),
                            0.0,
                        )
        if WRITE_CONSTANT:
            tl.store(d_constant_pointer + query_head, d_a)
        if WRITE_LINEAR:
            tl.store(d_linear_pointer + query_head * GMAX + groups, d_b)
        if WRITE_QUADRATIC:
            tl.store(
                d_quadratic_pointer + query_head * (GMAX * GMAX) + group_pairs, d_c
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
        raise TypeError("grouped causal prefix inputs must be tensors")
    if not triton_is_available():
        raise RuntimeError("grouped causal prefix kernels require Triton")
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
        raise ValueError("grouped causal prefix requires compatible rank-4 Q/K/V/G")
    if k.shape[1] != v.shape[1]:
        raise ValueError("grouped causal prefix requires matching K/V head counts")
    if not all(dimension > 0 for tensor in (q, k, v) for dimension in tensor.shape):
        raise ValueError("grouped causal prefix requires positive Q/K/V dimensions")
    if (
        q.dtype not in {torch.float16, torch.bfloat16}
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise TypeError(
            "grouped causal prefix kernels support matching float16/bfloat16 Q/K/V"
        )
    if grad_augmented.dtype != torch.float32:
        raise TypeError(
            "grouped causal prefix kernels require FP32 augmented gradients"
        )
    if q.shape[-1] != k.shape[-1] or not 1 <= q.shape[-1] <= 64:
        raise ValueError("grouped causal prefix kernels support D through 64")
    if not 1 <= v.shape[-1] <= 64:
        raise ValueError("grouped causal prefix kernels support DV through 64")
    if q.shape[1] % k.shape[1] != 0 or q.shape[1] // k.shape[1] > 16:
        raise ValueError(
            "grouped causal prefix kernels support native GQA ratios through 16"
        )
    if dim_groups.dtype != torch.int32:
        raise TypeError("dim_groups must be int32")
    if any(tensor.dtype != torch.float32 for tensor in (constant, linear, quadratic)):
        raise TypeError("grouped causal prefix coefficients must be FP32")
    gmax = linear.shape[-1] if linear.ndim == 2 else 0
    if (
        constant.shape != (q.shape[1],)
        or linear.shape != (q.shape[1], gmax)
        or quadratic.shape != (q.shape[1], gmax, gmax)
    ):
        raise ValueError(
            "grouped causal prefix requires expanded [Hq]/[Hq,G]/[Hq,G,G] coefficients"
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
        raise ValueError("grouped causal prefix tensors must share one device")
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
        raise RuntimeError("grouped causal prefix kernels require CUDA tensors")
    if not all(
        tensor.is_contiguous()
        for tensor in (q, k, v, dim_groups, grad_augmented, linear, quadratic)
    ):
        raise ValueError(
            "planned grouped causal prefix requires contiguous Q/K/V/G/groups/B/C"
        )
    if constant.ndim != 1 or constant.stride(0) <= 0:
        raise ValueError(
            "planned grouped causal prefix requires a positive-stride [Hq] A"
        )


def _coefficient_gradient_mask(
    need_coefficients: bool,
    coefficient_gradient_mask: tuple[bool, bool, bool] | None,
) -> tuple[bool, bool, bool]:
    if coefficient_gradient_mask is None:
        return (need_coefficients, need_coefficients, need_coefficients)
    normalized = tuple(coefficient_gradient_mask)
    if len(normalized) != 3 or not all(isinstance(value, bool) for value in normalized):
        raise TypeError("coefficient_gradient_mask must contain three A/B/C booleans")
    if any(normalized) != need_coefficients:
        raise ValueError("need_coefficients must agree with coefficient_gradient_mask")
    return normalized


def _validate_prefix_plan(
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
    """Fail closed unless the complete live plan payload is still identical."""
    from .grouped_quadratic_causal_backward import (
        _require_live_backward_prefix_plan_identity,
        _runtime_backward_prefix_geometry,
    )

    if not isinstance(plan, KernelPlan):
        raise TypeError("planned grouped causal prefix requires a KernelPlan")
    if plan.geometry.workspace_budget_bytes is None:
        raise ValueError("prefix kernel plan must carry a resolved workspace budget")
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
            "prefix kernel plan is stale for the live CUDA/tensor/toolchain geometry"
        )
    live_plan = _require_live_backward_prefix_plan_identity(plan, current_geometry)
    free_bytes, _ = torch.cuda.mem_get_info(q.device)
    if min(512 * 2**20, free_bytes // 20) < live_plan.geometry.workspace_budget_bytes:
        raise ValueError("live free memory cannot honor the prefix kernel plan")
    config = planned_grouped_causal_scan_config(live_plan)
    prefix_workspace_allocation_names(live_plan)
    return (
        config.token_block,
        config.pair_block,
        config.value_block,
    )


def grouped_causal_triton_prefix_dq_dcoeff(
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
    need_q: bool,
    need_coefficients: bool,
    coefficient_gradient_mask: tuple[bool, bool, bool] | None = None,
    requested_gradient_mask: tuple[bool, bool, bool, bool, bool, bool] | None = None,
    plan: KernelPlan | None = None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Run the exact planned H-prefix VJP with no allocator-external scratch."""
    if not need_q and not need_coefficients:
        return None, None, None, None
    _validate_triton_inputs(
        q, k, v, dim_groups, grad_augmented, constant, linear, quadratic
    )
    assert triton is not None
    needs_a, needs_b, needs_c = _coefficient_gradient_mask(
        need_coefficients,
        coefficient_gradient_mask,
    )
    requested_mask = (
        (need_q, False, False, needs_a, needs_b, needs_c)
        if requested_gradient_mask is None
        else tuple(requested_gradient_mask)
    )
    if len(requested_mask) != 6 or not all(
        isinstance(value, bool) for value in requested_mask
    ):
        raise TypeError("requested_gradient_mask must contain six Q/K/V/A/B/C booleans")
    if requested_mask[0] != need_q or requested_mask[3:] != (needs_a, needs_b, needs_c):
        raise ValueError("requested_gradient_mask must agree with prefix outputs")
    if plan is None:
        # Compatibility callers still receive the same strict internal plan;
        # the public path never reaches this launcher before admission.
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
    token_block, pair_block, value_block = _validate_prefix_plan(
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
    value_tiles = (augmented_dimension + value_block - 1) // value_block
    pair_groups = (pair_count + pair_block - 1) // pair_block
    batch_query_heads = batch_size * query_heads
    batch_key_value_heads = batch_size * key_value_heads
    d_pad = triton.next_power_of_2(head_dimension)
    groups_per_head = dim_groups.ndim == 2
    gqa = query_heads // key_value_heads
    workspace = allocate_planned_prefix_workspace(plan, device=q.device)
    h0, h1, h2 = _prefix_state_views(plan, workspace)
    placeholder = grad_augmented
    h0_dot_partials = workspace.get("prefix_h0_dot_partial", placeholder)
    h1_dot_partials = workspace.get("prefix_h1_dot_partial", placeholder)
    d_q_block = workspace.get("prefix_dq_macro", placeholder)
    d_a_macro = workspace.get("d_a_macro", placeholder)
    d_b_macro = workspace.get("d_b_macro", placeholder)
    h2_dot_partials = workspace.get("prefix_h2_dot_partial", placeholder)
    d_c_macro = workspace.get("d_c_macro", placeholder)
    d_q = workspace.get("dq", q)
    d_constant = workspace.get("d_constant", constant)
    d_linear = workspace.get("d_linear", linear)
    d_quadratic = workspace.get("d_quadratic", quadratic)
    d_a_macro_flat = (
        d_a_macro.reshape(batch_query_heads, token_block) if needs_a else placeholder
    )
    d_b_macro_flat = (
        d_b_macro.reshape(batch_query_heads, token_block, gmax)
        if needs_b
        else placeholder
    )
    d_c_macro_flat = (
        d_c_macro.reshape(batch_query_heads, token_block, gmax * gmax)
        if needs_c
        else placeholder
    )
    h0_dot_partials_flat = (
        h0_dot_partials.reshape(batch_query_heads, token_block, value_tiles)
        if needs_a
        else placeholder
    )
    h1_dot_partials_flat = (
        h1_dot_partials.reshape(
            batch_query_heads, token_block, value_tiles, head_dimension
        )
        if (need_q or needs_b)
        else placeholder
    )
    d_q_block_flat = (
        d_q_block.reshape(batch_query_heads, token_block, head_dimension)
        if need_q
        else placeholder
    )
    h2_dot_partials_flat = (
        h2_dot_partials.reshape(
            batch_query_heads, pair_groups, token_block, value_tiles, pair_block
        )
        if (need_q or needs_c)
        else placeholder
    )
    pair_rows = workspace.get("prefix_pair_rows", placeholder)
    pair_columns = workspace.get("prefix_pair_columns", placeholder)

    for chunk_index, chunk_start in enumerate(range(0, token_count, token_block)):
        chunk_length = min(token_block, token_count - chunk_start)
        if need_q or needs_a or needs_b:
            _prefix_h0_h1_token_block_kernel[(value_tiles, batch_key_value_heads)](
                k,
                v,
                grad_augmented,
                h0,
                h1,
                h0_dot_partials_flat,
                h1_dot_partials_flat,
                chunk_start,
                chunk_length,
                query_heads,
                key_value_heads,
                head_dimension,
                value_dimension,
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
                h0.stride(0),
                h0.stride(1),
                h1.stride(0),
                h1.stride(1),
                h1.stride(2),
                h0_dot_partials_flat.stride(0),
                h0_dot_partials_flat.stride(1),
                h1_dot_partials_flat.stride(0),
                h1_dot_partials_flat.stride(1),
                h1_dot_partials_flat.stride(2),
                D_PAD=d_pad,
                GQA=gqa,
                BT=token_block,
                BV=value_block,
                WRITE_CONSTANT=needs_a,
                WRITE_H1_DOT=need_q or needs_b,
                num_warps=4,
            )
            _reduce_prefix_h0_h1_kernel[(token_block, batch_query_heads)](
                q,
                dim_groups,
                linear,
                h0_dot_partials_flat,
                h1_dot_partials_flat,
                d_q_block_flat,
                d_a_macro_flat,
                d_b_macro_flat,
                chunk_start,
                chunk_length,
                query_heads,
                head_dimension,
                scale,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                h0_dot_partials_flat.stride(0),
                h0_dot_partials_flat.stride(1),
                h1_dot_partials_flat.stride(0),
                h1_dot_partials_flat.stride(1),
                h1_dot_partials_flat.stride(2),
                d_q_block_flat.stride(0),
                d_q_block_flat.stride(1),
                d_a_macro_flat.stride(0),
                d_a_macro_flat.stride(1),
                d_b_macro_flat.stride(0),
                d_b_macro_flat.stride(1),
                GROUPS_PER_HEAD=groups_per_head,
                GMAX=gmax,
                D_PAD=d_pad,
                VALUE_TILES=value_tiles,
                BT=token_block,
                WRITE_Q=need_q,
                WRITE_CONSTANT=needs_a,
                WRITE_LINEAR=needs_b,
                num_warps=4,
            )
        if need_q or needs_c:
            _prefix_h2_all_pair_kernel[
                (pair_groups, value_tiles, batch_key_value_heads)
            ](
                k,
                v,
                grad_augmented,
                pair_rows,
                pair_columns,
                h2,
                h2_dot_partials_flat,
                chunk_start,
                chunk_length,
                query_heads,
                key_value_heads,
                value_dimension,
                pair_count,
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
                h2.stride(0),
                h2.stride(1),
                h2.stride(2),
                h2_dot_partials_flat.stride(0),
                h2_dot_partials_flat.stride(1),
                h2_dot_partials_flat.stride(2),
                h2_dot_partials_flat.stride(3),
                GQA=gqa,
                BT=token_block,
                BP=pair_block,
                BV=value_block,
                num_warps=4,
            )
            _reduce_prefix_all_pair_and_accumulate_kernel[
                (token_block, batch_query_heads)
            ](
                q,
                dim_groups,
                quadratic,
                pair_rows,
                pair_columns,
                h2_dot_partials_flat,
                d_q_block_flat,
                d_c_macro_flat,
                chunk_start,
                chunk_length,
                query_heads,
                head_dimension,
                pair_count,
                scale,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                h2_dot_partials_flat.stride(0),
                h2_dot_partials_flat.stride(1),
                h2_dot_partials_flat.stride(2),
                h2_dot_partials_flat.stride(3),
                d_q_block_flat.stride(0),
                d_q_block_flat.stride(1),
                d_c_macro_flat.stride(0),
                d_c_macro_flat.stride(1),
                GROUPS_PER_HEAD=groups_per_head,
                GMAX=gmax,
                D_PAD=d_pad,
                VALUE_TILES=value_tiles,
                BT=token_block,
                BP=pair_block,
                PAIR_GROUPS=pair_groups,
                WRITE_Q=need_q,
                WRITE_COEFFICIENTS=needs_c,
                num_warps=4,
            )
        if need_q:
            _write_prefix_gradients_kernel[(token_block, batch_query_heads)](
                d_q_block_flat,
                d_q,
                chunk_start,
                chunk_length,
                query_heads,
                head_dimension,
                d_q_block_flat.stride(0),
                d_q_block_flat.stride(1),
                d_q.stride(0),
                d_q.stride(1),
                d_q.stride(2),
                d_q.stride(3),
                D_PAD=d_pad,
                BT=token_block,
                num_warps=4,
            )
        if needs_a or needs_b or needs_c:
            _coefficient_carry_kernel[(query_heads,)](
                d_a_macro_flat,
                d_b_macro_flat,
                d_c_macro_flat,
                d_constant,
                d_linear,
                d_quadratic,
                chunk_length,
                query_heads,
                d_a_macro_flat.stride(0),
                d_a_macro_flat.stride(1),
                d_b_macro_flat.stride(0),
                d_b_macro_flat.stride(1),
                d_c_macro_flat.stride(0),
                d_c_macro_flat.stride(1),
                GMAX=gmax,
                BATCH_SIZE=batch_size,
                BT=token_block,
                FIRST_CHUNK=chunk_index == 0,
                WRITE_CONSTANT=needs_a,
                WRITE_LINEAR=needs_b,
                WRITE_QUADRATIC=needs_c,
                num_warps=1,
            )
    _capture_prefix_production_witness(h2, plan=plan)
    return (
        d_q if need_q else None,
        d_constant if needs_a else None,
        d_linear if needs_b else None,
        d_quadratic if needs_c else None,
    )


__all__ = (
    "allocate_planned_prefix_workspace",
    "grouped_causal_triton_prefix_dq_dcoeff",
    "prefix_production_witness_capture_names",
    "prefix_workspace_allocation_names",
    "triton_is_available",
)
