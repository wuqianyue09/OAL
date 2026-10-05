"""Triton primitives for private HD Block-GEMM query feature materialization."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = None
    tl = None


_HEAD_DIMENSION = 64
_PAIR_COUNT = _HEAD_DIMENSION * (_HEAD_DIMENSION + 1) // 2
_FEATURE_DIMENSION = 1 + _HEAD_DIMENSION + _PAIR_COUNT
_FEATURES_PER_PROGRAM = 256
_FOLD_DIMENSIONS_PER_PROGRAM = 16
_FOLD_PARTNERS_PER_ITERATION = 8
_FOLD_PARTIAL_DIMENSIONS_PER_PROGRAM = 32
_FOLD_PARTIAL_PAIRS_PER_PROGRAM = 32
_MAX_TRITON_ARANGE_END = 1_048_576


def _query_fold_token_tile(token_block: int) -> int:
    """Return the legal power-of-two token tile for fold reductions."""
    if not isinstance(token_block, int) or isinstance(token_block, bool):
        raise TypeError("query-fold token block must be an integer")
    if token_block <= 0:
        raise ValueError("query-fold token block must be positive")
    tile = 1 << (token_block - 1).bit_length()
    if tile > _MAX_TRITON_ARANGE_END:
        raise ValueError(
            "query-fold token tile exceeds the maximum supported token tile"
        )
    return tile


def _query_feature_grid(
    batch_size: int,
    heads: int,
    token_count: int,
    token_tile: int,
) -> tuple[int, int]:
    """Return a head-local token-tile grid for query feature materialization."""
    if token_tile not in (1, 2, 4):
        raise ValueError("unsupported query feature token tile")
    token_tiles = (token_count + token_tile - 1) // token_tile
    feature_tiles = (_FEATURE_DIMENSION + _FEATURES_PER_PROGRAM - 1) // (
        _FEATURES_PER_PROGRAM
    )
    return batch_size * heads * token_tiles, feature_tiles


def triton_is_available() -> bool:
    """Return whether this private kernel module can launch Triton code."""

    return triton is not None and tl is not None


if triton_is_available():

    @triton.jit
    def _build_query_features_kernel(
        q_pointer,
        a_pointer,
        b_pointer,
        c_pointer,
        scale_pointer,
        pair_rows_pointer,
        pair_columns_pointer,
        pair_multiplicity_pointer,
        output_pointer,
        token_count,
        q_batch_stride,
        q_head_stride,
        q_token_stride,
        output_batch_stride,
        output_head_stride,
        output_token_stride,
        HEADS: tl.constexpr,
        D: tl.constexpr,
        PAIRS: tl.constexpr,
        FEATURES: tl.constexpr,
        BLOCK_FEATURES: tl.constexpr,
        TOKEN_TILE: tl.constexpr,
    ):
        flat_token_tile = tl.program_id(0)
        feature_offsets = tl.program_id(1) * BLOCK_FEATURES + tl.arange(
            0, BLOCK_FEATURES
        )
        token_tiles = (token_count + TOKEN_TILE - 1) // TOKEN_TILE
        token_tile = flat_token_tile % token_tiles
        head = (flat_token_tile // token_tiles) % HEADS
        batch = flat_token_tile // (token_tiles * HEADS)
        tokens = token_tile * TOKEN_TILE + tl.arange(0, TOKEN_TILE)
        token_mask = tokens < token_count
        output_mask = feature_offsets < FEATURES
        is_constant = feature_offsets == 0
        is_linear = (feature_offsets >= 1) & (feature_offsets <= D)
        is_pair = feature_offsets >= (D + 1)
        pair_offsets = feature_offsets - (D + 1)

        q_base = (
            q_pointer
            + batch * q_batch_stride
            + head * q_head_stride
            + tokens[:, None] * q_token_stride
        )
        output_base = (
            output_pointer
            + batch * output_batch_stride
            + head * output_head_stride
            + tokens[:, None] * output_token_stride
        )
        scale = tl.load(scale_pointer).to(tl.float32)
        constant = tl.load(a_pointer + head).to(tl.float32)
        linear_offsets = feature_offsets - 1
        linear_values = tl.load(
            q_base + linear_offsets[None, :],
            mask=token_mask[:, None] & is_linear[None, :],
            other=0.0,
        ).to(tl.float32)
        linear_values *= scale
        linear_coefficients = tl.load(
            b_pointer + head * D + linear_offsets,
            mask=is_linear,
            other=0.0,
        ).to(tl.float32)
        linear_values *= linear_coefficients

        pair_mask = is_pair & (pair_offsets < PAIRS)
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
            q_base + rows[None, :],
            mask=token_mask[:, None] & pair_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        column_values = tl.load(
            q_base + columns[None, :],
            mask=token_mask[:, None] & pair_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        row_values *= scale
        column_values *= scale
        pair_values = row_values * column_values
        pair_coefficients = tl.load(
            c_pointer + head * PAIRS + pair_offsets,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32)
        pair_values *= pair_coefficients
        multiplicity = tl.load(
            pair_multiplicity_pointer + pair_offsets,
            mask=pair_mask,
            other=0,
        ).to(tl.float32)
        pair_values *= multiplicity

        values = tl.where(
            token_mask[:, None] & is_constant[None, :],
            constant,
            0.0,
        )
        values = tl.where(is_linear[None, :], linear_values, values)
        values = tl.where(pair_mask[None, :], pair_values, values)
        tl.store(
            output_base + feature_offsets[None, :],
            values,
            mask=token_mask[:, None] & output_mask[None, :],
        )

    @triton.jit
    def _fold_query_gradient_kernel(
        q_pointer,
        d_phi_pointer,
        b_pointer,
        c_pointer,
        scale_pointer,
        d_q_pointer,
        q_batch_stride,
        q_head_stride,
        q_block_stride,
        q_token_stride,
        d_phi_batch_stride,
        d_phi_head_stride,
        d_phi_block_stride,
        d_phi_token_stride,
        d_q_batch_stride,
        d_q_head_stride,
        d_q_block_stride,
        d_q_token_stride,
        WAVE_BLOCKS: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
        D: tl.constexpr,
        PAIRS: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_PARTNERS: tl.constexpr,
        HEADS: tl.constexpr,
        TOKEN_GROUPS: tl.constexpr,
        DQ_TOKEN_TILE: tl.constexpr,
        Q_IS_RAW: tl.constexpr,
        VALID_TOKENS: tl.constexpr,
    ):
        flat_token_group = tl.program_id(0)
        dimension_tile = tl.program_id(1)
        token_group = flat_token_group % TOKEN_GROUPS
        wave_block = (flat_token_group // TOKEN_GROUPS) % WAVE_BLOCKS
        head = (flat_token_group // (TOKEN_GROUPS * WAVE_BLOCKS)) % HEADS
        batch = flat_token_group // (TOKEN_GROUPS * WAVE_BLOCKS * HEADS)
        dimensions = dimension_tile * BLOCK_D + tl.arange(0, BLOCK_D)
        scale = tl.load(scale_pointer).to(tl.float32)
        for token_offset in range(0, DQ_TOKEN_TILE):
            token = token_group * DQ_TOKEN_TILE + token_offset
            token_mask = (token < BLOCK_TOKENS) & (
                wave_block * BLOCK_TOKENS + token < VALID_TOKENS
            )
            dimension_mask = (dimensions < D) & token_mask
            q_base = (
                q_pointer
                + batch * q_batch_stride
                + head * q_head_stride
                + wave_block * q_block_stride
                + token * q_token_stride
            )
            d_phi_base = (
                d_phi_pointer
                + batch * d_phi_batch_stride
                + head * d_phi_head_stride
                + wave_block * d_phi_block_stride
                + token * d_phi_token_stride
            )
            d_q_base = (
                d_q_pointer
                + batch * d_q_batch_stride
                + head * d_q_head_stride
                + wave_block * d_q_block_stride
                + token * d_q_token_stride
            )
            linear_gradient = tl.load(
                d_phi_base + 1 + dimensions,
                mask=dimension_mask,
                other=0.0,
            ).to(tl.float32)
            linear_coefficient = tl.load(
                b_pointer + head * D + dimensions,
                mask=dimension_mask,
                other=0.0,
            ).to(tl.float32)
            d_q = linear_gradient * linear_coefficient * scale
            for partner_start in range(0, D, BLOCK_PARTNERS):
                partners = partner_start + tl.arange(0, BLOCK_PARTNERS)
                partner_mask = partners < D
                rows = tl.maximum(dimensions[:, None], partners[None, :])
                columns = tl.minimum(dimensions[:, None], partners[None, :])
                pair_offsets = rows * (rows + 1) // 2 + columns
                pair_mask = dimension_mask[:, None] & partner_mask[None, :]
                pair_gradient = tl.load(
                    d_phi_base + 1 + D + pair_offsets,
                    mask=pair_mask,
                    other=0.0,
                ).to(tl.float32)
                pair_coefficient = tl.load(
                    c_pointer + head * PAIRS + pair_offsets,
                    mask=pair_mask,
                    other=0.0,
                ).to(tl.float32)
                partner_values = tl.load(
                    q_base + partners,
                    mask=partner_mask & token_mask,
                    other=0.0,
                ).to(tl.float32)
                if Q_IS_RAW:
                    partner_values *= scale
                pair_multiplicity = tl.where(
                    rows == columns,
                    1.0,
                    2.0,
                )
                pair_contributions = (
                    pair_gradient
                    * pair_coefficient
                    * pair_multiplicity
                    * scale
                    * partner_values[None, :]
                )
                pair_contributions = tl.where(
                    rows == columns,
                    2.0 * pair_contributions,
                    pair_contributions,
                )
                d_q += tl.sum(pair_contributions, axis=1)
            tl.store(d_q_base + dimensions, d_q, mask=dimension_mask)

    @triton.jit
    def _fold_query_a_partials_kernel(
        d_phi_pointer,
        a_partials_pointer,
        block_start,
        d_phi_batch_stride,
        d_phi_head_stride,
        d_phi_block_stride,
        d_phi_token_stride,
        a_batch_stride,
        a_block_stride,
        a_head_stride,
        WAVE_BLOCKS: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
        TOKEN_TILE: tl.constexpr,
        HEADS: tl.constexpr,
    ):
        flat_block = tl.program_id(0)
        wave_block = flat_block % WAVE_BLOCKS
        head = (flat_block // WAVE_BLOCKS) % HEADS
        batch = flat_block // (WAVE_BLOCKS * HEADS)
        tokens = tl.arange(0, TOKEN_TILE)
        token_mask = tokens < BLOCK_TOKENS
        d_phi_base = (
            d_phi_pointer
            + batch * d_phi_batch_stride
            + head * d_phi_head_stride
            + wave_block * d_phi_block_stride
            + tokens * d_phi_token_stride
        )
        partial = tl.sum(
            tl.load(d_phi_base, mask=token_mask, other=0.0).to(tl.float32),
            axis=0,
        )
        output = (
            a_partials_pointer
            + batch * a_batch_stride
            + (block_start + wave_block) * a_block_stride
            + head * a_head_stride
        )
        tl.store(output, partial)

    @triton.jit
    def _fold_query_b_partials_kernel(
        q_pointer,
        d_phi_pointer,
        scale_pointer,
        b_partials_pointer,
        block_start,
        q_batch_stride,
        q_head_stride,
        q_block_stride,
        q_token_stride,
        d_phi_batch_stride,
        d_phi_head_stride,
        d_phi_block_stride,
        d_phi_token_stride,
        b_batch_stride,
        b_block_stride,
        b_head_stride,
        b_feature_stride,
        WAVE_BLOCKS: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
        TOKEN_TILE: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HEADS: tl.constexpr,
        Q_IS_RAW: tl.constexpr,
        VALID_TOKENS: tl.constexpr,
    ):
        flat_block = tl.program_id(0)
        feature_start = tl.program_id(1) * BLOCK_D
        wave_block = flat_block % WAVE_BLOCKS
        head = (flat_block // WAVE_BLOCKS) % HEADS
        batch = flat_block // (WAVE_BLOCKS * HEADS)
        tokens = tl.arange(0, TOKEN_TILE)
        token_mask = (tokens < BLOCK_TOKENS) & (
            wave_block * BLOCK_TOKENS + tokens < VALID_TOKENS
        )
        features = feature_start + tl.arange(0, BLOCK_D)
        feature_mask = features < D
        value_mask = token_mask[:, None] & feature_mask[None, :]
        q_base = (
            q_pointer
            + batch * q_batch_stride
            + head * q_head_stride
            + wave_block * q_block_stride
            + tokens[:, None] * q_token_stride
            + features[None, :]
        )
        d_phi_base = (
            d_phi_pointer
            + batch * d_phi_batch_stride
            + head * d_phi_head_stride
            + wave_block * d_phi_block_stride
            + tokens[:, None] * d_phi_token_stride
            + 1
            + features[None, :]
        )
        q_values = tl.load(q_base, mask=value_mask, other=0.0).to(tl.float32)
        if Q_IS_RAW:
            q_values *= tl.load(scale_pointer).to(tl.float32)
        partial = tl.sum(
            q_values * tl.load(d_phi_base, mask=value_mask, other=0.0).to(tl.float32),
            axis=0,
        )
        output = (
            b_partials_pointer
            + batch * b_batch_stride
            + (block_start + wave_block) * b_block_stride
            + head * b_head_stride
            + features * b_feature_stride
        )
        tl.store(output, partial, mask=feature_mask)

    @triton.jit
    def _fold_query_c_partials_kernel(
        q_pointer,
        d_phi_pointer,
        scale_pointer,
        pair_rows_pointer,
        pair_columns_pointer,
        pair_multiplicity_pointer,
        c_partials_pointer,
        block_start,
        q_batch_stride,
        q_head_stride,
        q_block_stride,
        q_token_stride,
        d_phi_batch_stride,
        d_phi_head_stride,
        d_phi_block_stride,
        d_phi_token_stride,
        c_batch_stride,
        c_block_stride,
        c_head_stride,
        c_feature_stride,
        WAVE_BLOCKS: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
        TOKEN_TILE: tl.constexpr,
        D: tl.constexpr,
        PAIRS: tl.constexpr,
        BLOCK_PAIRS: tl.constexpr,
        HEADS: tl.constexpr,
        Q_IS_RAW: tl.constexpr,
        VALID_TOKENS: tl.constexpr,
    ):
        flat_block = tl.program_id(0)
        pair_start = tl.program_id(1) * BLOCK_PAIRS
        wave_block = flat_block % WAVE_BLOCKS
        head = (flat_block // WAVE_BLOCKS) % HEADS
        batch = flat_block // (WAVE_BLOCKS * HEADS)
        tokens = tl.arange(0, TOKEN_TILE)
        token_mask = (tokens < BLOCK_TOKENS) & (
            wave_block * BLOCK_TOKENS + tokens < VALID_TOKENS
        )
        pair_offsets = pair_start + tl.arange(0, BLOCK_PAIRS)
        pair_mask = pair_offsets < PAIRS
        value_mask = token_mask[:, None] & pair_mask[None, :]
        rows = tl.load(pair_rows_pointer + pair_offsets, mask=pair_mask, other=0)
        columns = tl.load(
            pair_columns_pointer + pair_offsets,
            mask=pair_mask,
            other=0,
        )
        multiplicity = tl.load(
            pair_multiplicity_pointer + pair_offsets,
            mask=pair_mask,
            other=0,
        ).to(tl.float32)
        q_base = (
            q_pointer
            + batch * q_batch_stride
            + head * q_head_stride
            + wave_block * q_block_stride
            + tokens[:, None] * q_token_stride
        )
        d_phi_base = (
            d_phi_pointer
            + batch * d_phi_batch_stride
            + head * d_phi_head_stride
            + wave_block * d_phi_block_stride
            + tokens[:, None] * d_phi_token_stride
            + 1
            + D
            + pair_offsets[None, :]
        )
        row_values = tl.load(
            q_base + rows[None, :],
            mask=value_mask,
            other=0.0,
        ).to(tl.float32)
        column_values = tl.load(
            q_base + columns[None, :],
            mask=value_mask,
            other=0.0,
        ).to(tl.float32)
        if Q_IS_RAW:
            input_scale = tl.load(scale_pointer).to(tl.float32)
            row_values *= input_scale
            column_values *= input_scale
        partial = tl.sum(
            tl.load(d_phi_base, mask=value_mask, other=0.0).to(tl.float32)
            * row_values
            * column_values
            * multiplicity[None, :],
            axis=0,
        )
        output = (
            c_partials_pointer
            + batch * c_batch_stride
            + (block_start + wave_block) * c_block_stride
            + head * c_head_stride
            + pair_offsets * c_feature_stride
        )
        tl.store(output, partial, mask=pair_mask)


def build_query_features(
    q: torch.Tensor,
    output: torch.Tensor,
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    scale: torch.Tensor,
    pair_rows: torch.Tensor,
    pair_columns: torch.Tensor,
    pair_multiplicity: torch.Tensor,
    token_tile: int = 1,
) -> None:
    """Write the canonical D=64 query feature into a supplied BF16 buffer."""

    if not triton_is_available():
        raise RuntimeError("HD Block-GEMM query feature kernel requires Triton")
    if q.ndim != 4 or output.ndim != 4:
        raise ValueError(
            "HD Block-GEMM query feature kernel requires rank-four tensors"
        )
    token_count = q.shape[2]
    if token_count <= 0:
        raise ValueError("HD Block-GEMM query feature kernel requires positive tokens")
    if token_tile not in (1, 2, 4):
        raise ValueError("unsupported query feature token tile")
    assert triton is not None
    _build_query_features_kernel[
        _query_feature_grid(q.shape[0], q.shape[1], token_count, token_tile)
    ](
        q,
        a,
        b,
        c,
        scale,
        pair_rows,
        pair_columns,
        pair_multiplicity,
        output,
        token_count,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        HEADS=q.shape[1],
        D=_HEAD_DIMENSION,
        PAIRS=_PAIR_COUNT,
        FEATURES=_FEATURE_DIMENSION,
        BLOCK_FEATURES=_FEATURES_PER_PROGRAM,
        TOKEN_TILE=token_tile,
        num_warps=4,
    )


def fold_query_feature_gradient(
    q: torch.Tensor | None,
    d_phi_q: torch.Tensor,
    d_q: torch.Tensor | None,
    *,
    b: torch.Tensor | None,
    c: torch.Tensor | None,
    scale: torch.Tensor | None,
    pair_rows: torch.Tensor | None,
    pair_columns: torch.Tensor | None,
    pair_multiplicity: torch.Tensor | None,
    a_block_partials: torch.Tensor | None,
    b_block_partials: torch.Tensor | None,
    c_block_partials: torch.Tensor | None,
    block_start: int | None,
    q_is_raw: bool = False,
    valid_tokens: int | None = None,
    dq_token_tile: int = 1,
) -> None:
    """Fold a D=64 FP32 dPhiQ wave without global coefficient atomics."""

    if not triton_is_available():
        raise RuntimeError("HD Block-GEMM query fold kernel requires Triton")
    if d_phi_q.ndim != 5:
        raise ValueError("HD Block-GEMM query fold requires rank-five dPhiQ")
    batch_size, query_heads, wave_blocks, token_block, feature_dimension = d_phi_q.shape
    if feature_dimension != _FEATURE_DIMENSION or token_block <= 0 or wave_blocks <= 0:
        raise ValueError("HD Block-GEMM query fold has an invalid D=64 wave")
    partial_token_tile = _query_fold_token_tile(token_block)
    if valid_tokens is None:
        valid_tokens = wave_blocks * token_block
    if dq_token_tile not in (1, 2, 4):
        raise ValueError("unsupported query fold dQ token tile")
    if q is not None:
        q_block_stride = token_block * q.stride(2) if q_is_raw else q.stride(2)
        q_token_stride = q.stride(2) if q_is_raw else q.stride(3)
    assert triton is not None

    if d_q is not None:
        if q is None or b is None or c is None or scale is None:
            raise RuntimeError("dQ fold inputs were not initialized")
        _fold_query_gradient_kernel[
            (
                batch_size
                * query_heads
                * wave_blocks
                * triton.cdiv(token_block, dq_token_tile),
                triton.cdiv(_HEAD_DIMENSION, _FOLD_DIMENSIONS_PER_PROGRAM),
            )
        ](
            q,
            d_phi_q,
            b,
            c,
            scale,
            d_q,
            q.stride(0),
            q.stride(1),
            q_block_stride,
            q_token_stride,
            d_phi_q.stride(0),
            d_phi_q.stride(1),
            d_phi_q.stride(2),
            d_phi_q.stride(3),
            d_q.stride(0),
            d_q.stride(1),
            d_q.stride(2),
            d_q.stride(3),
            WAVE_BLOCKS=wave_blocks,
            BLOCK_TOKENS=token_block,
            D=_HEAD_DIMENSION,
            PAIRS=_PAIR_COUNT,
            BLOCK_D=_FOLD_DIMENSIONS_PER_PROGRAM,
            BLOCK_PARTNERS=_FOLD_PARTNERS_PER_ITERATION,
            HEADS=query_heads,
            TOKEN_GROUPS=triton.cdiv(token_block, dq_token_tile),
            DQ_TOKEN_TILE=dq_token_tile,
            Q_IS_RAW=q_is_raw,
            VALID_TOKENS=valid_tokens,
            num_warps=4,
        )

    if a_block_partials is not None:
        if block_start is None:
            raise RuntimeError("dA block partial range was not initialized")
        _fold_query_a_partials_kernel[(batch_size * query_heads * wave_blocks,)](
            d_phi_q,
            a_block_partials,
            block_start,
            d_phi_q.stride(0),
            d_phi_q.stride(1),
            d_phi_q.stride(2),
            d_phi_q.stride(3),
            a_block_partials.stride(0),
            a_block_partials.stride(1),
            a_block_partials.stride(2),
            WAVE_BLOCKS=wave_blocks,
            BLOCK_TOKENS=token_block,
            TOKEN_TILE=partial_token_tile,
            HEADS=query_heads,
            num_warps=4,
        )

    if b_block_partials is not None:
        if q is None or block_start is None:
            raise RuntimeError("dB block partial inputs were not initialized")
        _fold_query_b_partials_kernel[
            (
                batch_size * query_heads * wave_blocks,
                triton.cdiv(_HEAD_DIMENSION, _FOLD_PARTIAL_DIMENSIONS_PER_PROGRAM),
            )
        ](
            q,
            d_phi_q,
            scale,
            b_block_partials,
            block_start,
            q.stride(0),
            q.stride(1),
            q_block_stride,
            q_token_stride,
            d_phi_q.stride(0),
            d_phi_q.stride(1),
            d_phi_q.stride(2),
            d_phi_q.stride(3),
            b_block_partials.stride(0),
            b_block_partials.stride(1),
            b_block_partials.stride(2),
            b_block_partials.stride(3),
            WAVE_BLOCKS=wave_blocks,
            BLOCK_TOKENS=token_block,
            TOKEN_TILE=partial_token_tile,
            D=_HEAD_DIMENSION,
            BLOCK_D=_FOLD_PARTIAL_DIMENSIONS_PER_PROGRAM,
            HEADS=query_heads,
            Q_IS_RAW=q_is_raw,
            VALID_TOKENS=valid_tokens,
            num_warps=4,
        )

    if c_block_partials is not None:
        if (
            q is None
            or pair_rows is None
            or pair_columns is None
            or pair_multiplicity is None
            or block_start is None
        ):
            raise RuntimeError("dC block partial inputs were not initialized")
        _fold_query_c_partials_kernel[
            (
                batch_size * query_heads * wave_blocks,
                triton.cdiv(_PAIR_COUNT, _FOLD_PARTIAL_PAIRS_PER_PROGRAM),
            )
        ](
            q,
            d_phi_q,
            scale,
            pair_rows,
            pair_columns,
            pair_multiplicity,
            c_block_partials,
            block_start,
            q.stride(0),
            q.stride(1),
            q_block_stride,
            q_token_stride,
            d_phi_q.stride(0),
            d_phi_q.stride(1),
            d_phi_q.stride(2),
            d_phi_q.stride(3),
            c_block_partials.stride(0),
            c_block_partials.stride(1),
            c_block_partials.stride(2),
            c_block_partials.stride(3),
            WAVE_BLOCKS=wave_blocks,
            BLOCK_TOKENS=token_block,
            TOKEN_TILE=partial_token_tile,
            D=_HEAD_DIMENSION,
            PAIRS=_PAIR_COUNT,
            BLOCK_PAIRS=_FOLD_PARTIAL_PAIRS_PER_PROGRAM,
            HEADS=query_heads,
            Q_IS_RAW=q_is_raw,
            VALID_TOKENS=valid_tokens,
            num_warps=4,
        )


__all__ = (
    "build_query_features",
    "fold_query_feature_gradient",
    "triton_is_available",
)
