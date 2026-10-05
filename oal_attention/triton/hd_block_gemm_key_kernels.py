"""Triton primitives for private HD Block-GEMM key feature operators."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = None
    tl = None


_FEATURES_PER_PROGRAM = 256
_FOLD_DIMENSIONS_PER_PROGRAM = 16
_FOLD_PARTNERS_PER_STEP = 8


def triton_is_available() -> bool:
    """Return whether this private kernel module can launch Triton code."""
    return triton is not None and tl is not None


if triton_is_available():

    @triton.jit
    def _build_key_features_kernel(
        key_pointer,
        pair_rows_pointer,
        pair_columns_pointer,
        output_pointer,
        token_count,
        key_batch_stride,
        key_head_stride,
        key_token_stride,
        output_batch_stride,
        output_head_stride,
        output_token_stride,
        HEADS: tl.constexpr,
        D: tl.constexpr,
        PAIRS: tl.constexpr,
        FEATURES: tl.constexpr,
        BLOCK_FEATURES: tl.constexpr,
    ):
        flat_token = tl.program_id(0)
        feature_offsets = tl.program_id(1) * BLOCK_FEATURES + tl.arange(
            0, BLOCK_FEATURES
        )
        head = (flat_token // token_count) % HEADS
        batch = flat_token // (token_count * HEADS)
        token = flat_token % token_count
        output_mask = feature_offsets < FEATURES
        is_constant = feature_offsets == 0
        is_linear = (feature_offsets >= 1) & (feature_offsets <= D)
        pair_offsets = feature_offsets - (D + 1)
        pair_mask = (pair_offsets >= 0) & (pair_offsets < PAIRS)

        key_base = (
            key_pointer
            + batch * key_batch_stride
            + head * key_head_stride
            + token * key_token_stride
        )
        output_base = (
            output_pointer
            + batch * output_batch_stride
            + head * output_head_stride
            + token * output_token_stride
        )
        linear_offsets = feature_offsets - 1
        linear_values = tl.load(
            key_base + linear_offsets,
            mask=is_linear,
            other=0.0,
        ).to(tl.float32)
        rows = tl.load(
            pair_rows_pointer + pair_offsets,
            mask=pair_mask,
            other=0,
        )
        columns = tl.load(
            pair_columns_pointer + pair_offsets,
            mask=pair_mask,
            other=0,
        )
        row_values = tl.load(
            key_base + rows,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32)
        column_values = tl.load(
            key_base + columns,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32)
        values = tl.where(is_constant, 1.0, 0.0)
        values = tl.where(is_linear, linear_values, values)
        values = tl.where(pair_mask, row_values * column_values, values)
        tl.store(output_base + feature_offsets, values, mask=output_mask)

    @triton.jit
    def _fold_key_feature_gradient_kernel(
        key_pointer,
        d_phi_pointer,
        output_pointer,
        token_count,
        key_batch_stride,
        key_head_stride,
        key_token_stride,
        d_phi_batch_stride,
        d_phi_head_stride,
        d_phi_token_stride,
        output_batch_stride,
        output_head_stride,
        output_token_stride,
        HEADS: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_PARTNERS: tl.constexpr,
    ):
        flat_token = tl.program_id(0)
        dimensions = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
        dimension_mask = dimensions < D
        head = (flat_token // token_count) % HEADS
        batch = flat_token // (token_count * HEADS)
        token = flat_token % token_count
        key_base = (
            key_pointer
            + batch * key_batch_stride
            + head * key_head_stride
            + token * key_token_stride
        )
        d_phi_base = (
            d_phi_pointer
            + batch * d_phi_batch_stride
            + head * d_phi_head_stride
            + token * d_phi_token_stride
        )
        output_base = (
            output_pointer
            + batch * output_batch_stride
            + head * output_head_stride
            + token * output_token_stride
        )
        folded = tl.load(
            d_phi_base + 1 + dimensions,
            mask=dimension_mask,
            other=0.0,
        ).to(tl.float32)
        dimension_grid = dimensions[:, None]
        for partner_start in tl.static_range(0, D, BLOCK_PARTNERS):
            partners = partner_start + tl.arange(0, BLOCK_PARTNERS)
            partner_mask = partners < D
            partner_grid = partners[None, :]
            row = tl.maximum(dimension_grid, partner_grid)
            column = tl.minimum(dimension_grid, partner_grid)
            pair_offset = row * (row + 1) // 2 + column
            pair_gradient = tl.load(
                d_phi_base + 1 + D + pair_offset,
                mask=dimension_mask[:, None] & partner_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            partner_value = tl.load(
                key_base + partner_grid,
                mask=partner_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            endpoint_count = tl.where(dimension_grid == partner_grid, 2.0, 1.0)
            folded += tl.sum(
                pair_gradient * partner_value * endpoint_count,
                axis=1,
            )
        tl.store(output_base + dimensions, folded, mask=dimension_mask)


def build_key_features(
    key: torch.Tensor,
    output: torch.Tensor,
    *,
    pair_rows: torch.Tensor,
    pair_columns: torch.Tensor,
    head_dimension: int,
    pair_count: int,
    feature_dimension: int,
) -> None:
    """Write one canonical key-feature wave into a supplied BF16 buffer."""
    if not triton_is_available():
        raise RuntimeError("HD Block-GEMM key feature kernel requires Triton")
    if key.ndim != 4 or output.ndim != 4:
        raise ValueError("HD Block-GEMM key feature kernel requires rank-four tensors")
    token_count = key.shape[2]
    if token_count <= 0:
        raise ValueError("HD Block-GEMM key feature kernel requires positive tokens")
    if pair_count != head_dimension * (head_dimension + 1) // 2:
        raise ValueError("HD Block-GEMM key feature pair count is inconsistent")
    if feature_dimension != 1 + head_dimension + pair_count:
        raise ValueError("HD Block-GEMM key feature dimension is inconsistent")
    assert triton is not None
    _build_key_features_kernel[
        (
            key.shape[0] * key.shape[1] * token_count,
            triton.cdiv(feature_dimension, _FEATURES_PER_PROGRAM),
        )
    ](
        key,
        pair_rows,
        pair_columns,
        output,
        token_count,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        HEADS=key.shape[1],
        D=head_dimension,
        PAIRS=pair_count,
        FEATURES=feature_dimension,
        BLOCK_FEATURES=_FEATURES_PER_PROGRAM,
        num_warps=4,
    )


def fold_key_feature_gradient(
    key: torch.Tensor,
    d_phi_key: torch.Tensor,
    output: torch.Tensor,
    *,
    head_dimension: int,
    pair_count: int,
    feature_dimension: int,
) -> None:
    """Fold one canonical key-feature gradient directly by output dimension."""
    if not triton_is_available():
        raise RuntimeError("HD Block-GEMM key fold kernel requires Triton")
    if key.ndim != 4 or d_phi_key.ndim != 4 or output.ndim != 4:
        raise ValueError("HD Block-GEMM key fold kernel requires rank-four tensors")
    token_count = key.shape[2]
    if token_count <= 0:
        raise ValueError("HD Block-GEMM key fold kernel requires positive tokens")
    if pair_count != head_dimension * (head_dimension + 1) // 2:
        raise ValueError("HD Block-GEMM key fold pair count is inconsistent")
    if feature_dimension != 1 + head_dimension + pair_count:
        raise ValueError("HD Block-GEMM key fold feature dimension is inconsistent")
    assert triton is not None
    _fold_key_feature_gradient_kernel[
        (
            key.shape[0] * key.shape[1] * token_count,
            triton.cdiv(head_dimension, _FOLD_DIMENSIONS_PER_PROGRAM),
        )
    ](
        key,
        d_phi_key,
        output,
        token_count,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        d_phi_key.stride(0),
        d_phi_key.stride(1),
        d_phi_key.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        HEADS=key.shape[1],
        D=head_dimension,
        BLOCK_D=_FOLD_DIMENSIONS_PER_PROGRAM,
        BLOCK_PARTNERS=_FOLD_PARTNERS_PER_STEP,
        num_warps=4,
    )


__all__ = (
    "build_key_features",
    "fold_key_feature_gradient",
    "triton_is_available",
)
