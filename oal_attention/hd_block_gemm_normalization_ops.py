"""Strict private normalization facade for HD Block-GEMM."""

from __future__ import annotations

import math

import torch

from .hd_block_gemm_plan import HDParallelBlockPlan
from .triton import hd_block_gemm_normalization_kernels


def _reference_forward_normalize(
    y: torch.Tensor,
    *,
    value_dimension: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized output while preserving raw FP32 auxiliary values."""
    if not isinstance(y, torch.Tensor) or y.ndim < 1 or y.dtype != torch.float32:
        raise TypeError("y must be an FP32 tensor")
    if value_dimension <= 0 or y.shape[-1] != value_dimension + 1:
        raise ValueError("value_dimension does not match y")
    if not isinstance(eps, float) or not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a positive finite float")
    numerator = y[..., :value_dimension].clone()
    denominator = y[..., value_dimension:].clone()
    output = numerator / denominator.clamp_min(eps)
    return output, numerator, denominator


def _reference_backward_normalize(
    grad_output: torch.Tensor | None,
    grad_numerator: torch.Tensor | None,
    grad_denominator: torch.Tensor | None,
    *,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    value_dimension: int,
    eps: float,
) -> torch.Tensor:
    """Return the existing augmented-normalization VJP in FP32."""
    if numerator.dtype != torch.float32 or numerator.shape[-1] != value_dimension:
        raise ValueError("numerator does not match value_dimension")
    if denominator.dtype != torch.float32 or denominator.shape != (
        *numerator.shape[:-1],
        1,
    ):
        raise ValueError("denominator does not match numerator")
    if not isinstance(eps, float) or not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a positive finite float")
    divisor = denominator.clamp_min(eps)
    value_gradient = torch.zeros_like(numerator)
    denominator_gradient = torch.zeros_like(denominator)
    if grad_output is not None:
        output_gradient = grad_output.float()
        value_gradient.copy_(output_gradient).div_(divisor)
        denominator_gradient.copy_(
            (output_gradient * numerator).sum(
                dim=-1,
                keepdim=True,
                dtype=torch.float32,
            )
        )
        denominator_gradient.div_(divisor).div_(divisor)
        denominator_gradient.neg_().mul_(denominator >= eps)
    if grad_numerator is not None:
        value_gradient.add_(grad_numerator)
    if grad_denominator is not None:
        denominator_gradient.add_(grad_denominator)
    return torch.cat((value_gradient, denominator_gradient), dim=-1)


def _stage_backward_denominator_dot(
    grad_output: torch.Tensor,
    numerator: torch.Tensor,
    *,
    output: torch.Tensor,
    value_dimension: int,
) -> None:
    """Reproduce the existing FP32 product and PyTorch reduction in-place."""
    value_product = output[..., :value_dimension]
    denominator_dot = output[..., value_dimension:]
    value_product.copy_(grad_output)
    value_product.mul_(numerator)
    torch.sum(
        value_product,
        dim=-1,
        keepdim=True,
        dtype=torch.float32,
        out=denominator_dot,
    )


def _require_dense_last_two(value: object, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.layout != torch.strided or value.stride(-1) != 1:
        raise ValueError(f"{name} must have a dense final dimension")
    if value.stride(-2) != value.shape[-1]:
        raise ValueError(f"{name} must have a dense token stride")
    return value


def _require_strided(value: object, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.layout != torch.strided:
        raise ValueError(f"{name} must use strided layout")
    return value


def forward_normalize_triton(
    y_wave: object,
    output: object,
    numerator: object,
    denominator: object,
    *,
    block_start: int,
    valid_tokens: int,
    eps: float,
    plan: object,
) -> None:
    """Normalize one query wave and preserve its raw numerator/denominator."""
    if not isinstance(plan, HDParallelBlockPlan):
        raise TypeError("plan must be an HDParallelBlockPlan")
    if plan.forward_normalize_impl != "triton":
        raise ValueError("Triton normalization requires forward_normalize_impl=triton")
    if (
        not isinstance(block_start, int)
        or isinstance(block_start, bool)
        or block_start < 0
    ):
        raise ValueError("block_start must be a non-negative integer")
    capacity = plan.feature_wave_blocks * plan.token_block
    if not isinstance(valid_tokens, int) or isinstance(valid_tokens, bool):
        raise TypeError("valid_tokens must be an integer")
    if valid_tokens <= 0 or valid_tokens > capacity:
        raise ValueError("valid_tokens exceeds the planned wave")
    if not isinstance(eps, float) or not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a positive finite float")

    device = torch.device(plan.device)
    y_wave = _require_dense_last_two(y_wave, name="y_wave")
    output = _require_dense_last_two(output, name="output")
    numerator = _require_dense_last_two(numerator, name="numerator")
    denominator = _require_dense_last_two(denominator, name="denominator")
    expected_y_shape = (
        plan.batch_size,
        plan.query_heads,
        capacity,
        plan.augmented_value_dimension,
    )
    if tuple(y_wave.shape) != expected_y_shape or y_wave.dtype != torch.float32:
        raise ValueError("y_wave does not match the FP32 plan workspace")
    expected_output_shape = (
        plan.batch_size,
        plan.query_heads,
        plan.sequence_length,
        plan.value_dimension,
    )
    if tuple(output.shape) != expected_output_shape:
        raise ValueError("output shape does not match the plan")
    if (
        tuple(numerator.shape) != expected_output_shape
        or numerator.dtype != torch.float32
    ):
        raise ValueError("numerator does not match the FP32 plan output")
    expected_denominator_shape = (*expected_output_shape[:-1], 1)
    if (
        tuple(denominator.shape) != expected_denominator_shape
        or denominator.dtype != torch.float32
    ):
        raise ValueError("denominator does not match the FP32 plan output")
    expected_output_dtype = getattr(torch, plan.buffer("output").dtype)
    if output.dtype != expected_output_dtype:
        raise ValueError("output dtype does not match the plan")
    tensors = (y_wave, output, numerator, denominator)
    if any(tensor.device != device for tensor in tensors):
        raise ValueError("normalization tensors must match the plan device")
    if not y_wave.is_cuda:
        raise RuntimeError("Triton forward normalization requires CUDA tensors")
    storage_pointers = {tensor.untyped_storage().data_ptr() for tensor in tensors}
    if len(storage_pointers) != len(tensors):
        raise ValueError("normalization tensors must not overlap")
    token_start = block_start * plan.token_block
    if token_start + valid_tokens > plan.sequence_length:
        raise ValueError("normalization wave exceeds the sequence length")
    if not hd_block_gemm_normalization_kernels.triton_is_available():
        raise RuntimeError("Triton forward normalization requires Triton")
    hd_block_gemm_normalization_kernels.forward_normalize(
        y_wave,
        output,
        numerator,
        denominator,
        block_start=block_start,
        valid_tokens=valid_tokens,
        token_block=plan.token_block,
        eps=eps,
        value_dimension=plan.value_dimension,
    )


def backward_normalize_triton(
    grad_output: object,
    grad_numerator: object,
    grad_denominator: object,
    numerator: object,
    denominator: object,
    output: object,
    *,
    eps: float,
    plan: object,
) -> None:
    """Write the fused FP32 augmented-normalization VJP."""
    if not isinstance(plan, HDParallelBlockPlan):
        raise TypeError("plan must be an HDParallelBlockPlan")
    if plan.backward_normalize_impl != "triton":
        raise ValueError("Triton normalization requires backward_normalize_impl=triton")
    output_shape = (
        plan.batch_size,
        plan.query_heads,
        plan.sequence_length,
        plan.value_dimension,
    )
    denominator_shape = (*output_shape[:-1], 1)
    augmented_shape = (*output_shape[:-1], plan.augmented_value_dimension)
    for value, name, shape, dtype in (
        (numerator, "numerator", output_shape, torch.float32),
        (denominator, "denominator", denominator_shape, torch.float32),
        (output, "output", augmented_shape, torch.float32),
    ):
        tensor = _require_dense_last_two(value, name=name)
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(f"{name} does not match the normalization plan")
    device = torch.device(plan.device)
    optional = (
        (grad_output, "grad_output", output_shape, getattr(torch, plan.output_dtype)),
        (grad_numerator, "grad_numerator", output_shape, torch.float32),
        (grad_denominator, "grad_denominator", denominator_shape, torch.float32),
    )
    tensors = [numerator, denominator, output]
    for value, name, shape, dtype in optional:
        if value is None:
            continue
        tensor = _require_strided(value, name=name)
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(f"{name} does not match the normalization plan")
        tensors.append(tensor)
    if any(tensor.device != device for tensor in tensors):
        raise ValueError("normalization tensors must match the plan device")
    if not output.is_cuda:
        raise RuntimeError("Triton backward normalization requires CUDA tensors")
    output_storage = output.untyped_storage().data_ptr()
    if any(
        tensor.untyped_storage().data_ptr() == output_storage
        for tensor in tensors
        if tensor is not output
    ):
        raise ValueError("normalization output must not overlap its inputs")
    if not isinstance(eps, float) or not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a positive finite float")
    if not hd_block_gemm_normalization_kernels.triton_is_available():
        raise RuntimeError("Triton backward normalization requires Triton")
    if grad_output is not None:
        _stage_backward_denominator_dot(
            grad_output,
            numerator,
            output=output,
            value_dimension=plan.value_dimension,
        )
    hd_block_gemm_normalization_kernels.backward_normalize(
        grad_output,
        grad_numerator,
        grad_denominator,
        numerator,
        denominator,
        output,
        eps=eps,
        value_dimension=plan.value_dimension,
    )


__all__ = ("backward_normalize_triton", "forward_normalize_triton")
