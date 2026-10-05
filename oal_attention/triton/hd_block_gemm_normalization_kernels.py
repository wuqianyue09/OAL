"""Triton normalization primitives for private HD Block-GEMM."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = None
    tl = None


_ROWS_PER_PROGRAM = 4


def triton_is_available() -> bool:
    return triton is not None and tl is not None


if triton_is_available():

    @triton.jit
    def _forward_normalize_kernel(
        y_pointer,
        output_pointer,
        numerator_pointer,
        denominator_pointer,
        token_start,
        valid_tokens,
        row_count,
        eps,
        y_batch_stride,
        y_head_stride,
        y_token_stride,
        output_batch_stride,
        output_head_stride,
        output_token_stride,
        numerator_batch_stride,
        numerator_head_stride,
        numerator_token_stride,
        denominator_batch_stride,
        denominator_head_stride,
        denominator_token_stride,
        HEADS: tl.constexpr,
        VALUE_DIMENSION: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_VALUE: tl.constexpr,
    ):
        flat_tokens = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        row_mask = flat_tokens < row_count
        value_offsets = tl.arange(0, BLOCK_VALUE)
        value_mask = value_offsets < VALUE_DIMENSION
        head = (flat_tokens // valid_tokens) % HEADS
        batch = flat_tokens // (valid_tokens * HEADS)
        local_token = flat_tokens % valid_tokens
        global_token = token_start + local_token

        y_base = (
            y_pointer
            + batch * y_batch_stride
            + head * y_head_stride
            + local_token * y_token_stride
        )
        output_base = (
            output_pointer
            + batch * output_batch_stride
            + head * output_head_stride
            + global_token * output_token_stride
        )
        numerator_base = (
            numerator_pointer
            + batch * numerator_batch_stride
            + head * numerator_head_stride
            + global_token * numerator_token_stride
        )
        denominator_address = (
            denominator_pointer
            + batch * denominator_batch_stride
            + head * denominator_head_stride
            + global_token * denominator_token_stride
        )
        raw_denominator = tl.load(
            y_base + VALUE_DIMENSION,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        divisor = tl.maximum(raw_denominator, eps)
        raw_numerator = tl.load(
            y_base[:, None] + value_offsets[None, :],
            mask=row_mask[:, None] & value_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        value_store_mask = row_mask[:, None] & value_mask[None, :]
        tl.store(
            numerator_base[:, None] + value_offsets[None, :],
            raw_numerator,
            mask=value_store_mask,
        )
        tl.store(denominator_address, raw_denominator, mask=row_mask)
        tl.store(
            output_base[:, None] + value_offsets[None, :],
            tl.div_rn(raw_numerator, divisor[:, None]),
            mask=value_store_mask,
        )

    @triton.jit
    def _backward_normalize_kernel(
        grad_output_pointer,
        grad_numerator_pointer,
        grad_denominator_pointer,
        denominator_pointer,
        output_pointer,
        row_count,
        eps,
        grad_output_batch_stride,
        grad_output_head_stride,
        grad_output_token_stride,
        grad_output_value_stride,
        grad_numerator_batch_stride,
        grad_numerator_head_stride,
        grad_numerator_token_stride,
        grad_numerator_value_stride,
        grad_denominator_batch_stride,
        grad_denominator_head_stride,
        grad_denominator_token_stride,
        denominator_batch_stride,
        denominator_head_stride,
        denominator_token_stride,
        output_batch_stride,
        output_head_stride,
        output_token_stride,
        HEADS: tl.constexpr,
        SEQUENCE_LENGTH: tl.constexpr,
        VALUE_DIMENSION: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_VALUE: tl.constexpr,
        HAS_GRAD_OUTPUT: tl.constexpr,
        HAS_GRAD_NUMERATOR: tl.constexpr,
        HAS_GRAD_DENOMINATOR: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        row_mask = rows < row_count
        values = tl.arange(0, BLOCK_VALUE)
        value_mask = values < VALUE_DIMENSION
        matrix_mask = row_mask[:, None] & value_mask[None, :]
        token = rows % SEQUENCE_LENGTH
        head = (rows // SEQUENCE_LENGTH) % HEADS
        batch = rows // (SEQUENCE_LENGTH * HEADS)
        denominator_address = (
            denominator_pointer
            + batch * denominator_batch_stride
            + head * denominator_head_stride
            + token * denominator_token_stride
        )
        denominator = tl.load(
            denominator_address,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        divisor = tl.maximum(denominator, eps)
        output_base = (
            output_pointer
            + batch * output_batch_stride
            + head * output_head_stride
            + token * output_token_stride
        )
        if HAS_GRAD_OUTPUT:
            grad_output_base = (
                grad_output_pointer
                + batch * grad_output_batch_stride
                + head * grad_output_head_stride
                + token * grad_output_token_stride
            )
            grad_output = tl.load(
                grad_output_base[:, None] + values[None, :] * grad_output_value_stride,
                mask=matrix_mask,
                other=0.0,
            ).to(tl.float32)
            value_gradient = tl.div_rn(grad_output, divisor[:, None])
            dot = tl.load(output_base + VALUE_DIMENSION, mask=row_mask, other=0.0)
            denominator_gradient = -tl.div_rn(
                tl.div_rn(dot, divisor),
                divisor,
            ) * (denominator >= eps)
        else:
            value_gradient = tl.zeros((BLOCK_ROWS, BLOCK_VALUE), tl.float32)
            denominator_gradient = tl.zeros((BLOCK_ROWS,), tl.float32)
        if HAS_GRAD_NUMERATOR:
            grad_numerator_base = (
                grad_numerator_pointer
                + batch * grad_numerator_batch_stride
                + head * grad_numerator_head_stride
                + token * grad_numerator_token_stride
            )
            value_gradient += tl.load(
                grad_numerator_base[:, None]
                + values[None, :] * grad_numerator_value_stride,
                mask=matrix_mask,
                other=0.0,
            ).to(tl.float32)
        if HAS_GRAD_DENOMINATOR:
            grad_denominator_address = (
                grad_denominator_pointer
                + batch * grad_denominator_batch_stride
                + head * grad_denominator_head_stride
                + token * grad_denominator_token_stride
            )
            denominator_gradient += tl.load(
                grad_denominator_address,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
        tl.store(
            output_base[:, None] + values[None, :],
            value_gradient,
            mask=matrix_mask,
        )
        tl.store(
            output_base + VALUE_DIMENSION,
            denominator_gradient,
            mask=row_mask,
        )


def forward_normalize(
    y_wave: torch.Tensor,
    output: torch.Tensor,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    *,
    block_start: int,
    valid_tokens: int,
    token_block: int,
    eps: float,
    value_dimension: int,
) -> None:
    if not triton_is_available():
        raise RuntimeError("HD Block-GEMM forward normalization requires Triton")
    assert triton is not None
    token_start = block_start * token_block
    block_value = triton.next_power_of_2(value_dimension)
    total_rows = y_wave.shape[0] * y_wave.shape[1] * valid_tokens
    _forward_normalize_kernel[(triton.cdiv(total_rows, _ROWS_PER_PROGRAM),)](
        y_wave,
        output,
        numerator,
        denominator,
        token_start,
        valid_tokens,
        total_rows,
        eps,
        y_wave.stride(0),
        y_wave.stride(1),
        y_wave.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        numerator.stride(0),
        numerator.stride(1),
        numerator.stride(2),
        denominator.stride(0),
        denominator.stride(1),
        denominator.stride(2),
        HEADS=y_wave.shape[1],
        VALUE_DIMENSION=value_dimension,
        BLOCK_ROWS=_ROWS_PER_PROGRAM,
        BLOCK_VALUE=block_value,
        num_warps=4,
    )


def backward_normalize(
    grad_output: torch.Tensor | None,
    grad_numerator: torch.Tensor | None,
    grad_denominator: torch.Tensor | None,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    output: torch.Tensor,
    *,
    eps: float,
    value_dimension: int,
) -> None:
    if not triton_is_available():
        raise RuntimeError("HD Block-GEMM backward normalization requires Triton")
    assert triton is not None
    row_count = numerator.numel() // value_dimension
    block_value = triton.next_power_of_2(value_dimension)
    fallback = numerator
    _backward_normalize_kernel[(triton.cdiv(row_count, _ROWS_PER_PROGRAM),)](
        grad_output if grad_output is not None else fallback,
        grad_numerator if grad_numerator is not None else fallback,
        grad_denominator if grad_denominator is not None else denominator,
        denominator,
        output,
        row_count,
        eps,
        *(grad_output.stride() if grad_output is not None else (0, 0, 0, 0)),
        *(grad_numerator.stride() if grad_numerator is not None else (0, 0, 0, 0)),
        *(grad_denominator.stride()[:3] if grad_denominator is not None else (0, 0, 0)),
        *denominator.stride()[:3],
        *output.stride()[:3],
        HEADS=numerator.shape[1],
        SEQUENCE_LENGTH=numerator.shape[2],
        VALUE_DIMENSION=value_dimension,
        BLOCK_ROWS=_ROWS_PER_PROGRAM,
        BLOCK_VALUE=block_value,
        HAS_GRAD_OUTPUT=grad_output is not None,
        HAS_GRAD_NUMERATOR=grad_numerator is not None,
        HAS_GRAD_DENOMINATOR=grad_denominator is not None,
        num_warps=4,
    )


__all__ = ("backward_normalize", "forward_normalize", "triton_is_available")
