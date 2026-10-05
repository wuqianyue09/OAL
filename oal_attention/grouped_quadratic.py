"""Public grouped quadratic attention API."""

from __future__ import annotations

import math
from typing import Literal

import torch

from .grouped_reference import grouped_quadratic_reference

_SUPPORTED_FLOATING_DTYPES = frozenset(
    {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
)
_SUPPORTED_FACTOR_DTYPES = frozenset({torch.float32, torch.float64})
_SUPPORTED_OUTPUT_DTYPES = _SUPPORTED_FLOATING_DTYPES


def _resolve_working_dtype(
    q_dtype: torch.dtype,
    factor_dtype: torch.dtype,
) -> torch.dtype:
    promoted = torch.promote_types(q_dtype, factor_dtype)
    return torch.float64 if promoted == torch.float64 else torch.float32


def _validate_kernel_eps(
    kernel_eps: object,
    *,
    working_dtype: torch.dtype,
) -> float:
    if isinstance(kernel_eps, bool) or not isinstance(kernel_eps, (int, float)):
        raise TypeError("kernel_eps must be a positive finite Python scalar")
    try:
        resolved = float(kernel_eps)
    except OverflowError:
        raise ValueError("kernel_eps must be finite") from None
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError("kernel_eps must be finite and positive")
    limits = torch.finfo(working_dtype)
    smallest_subnormal = limits.tiny * limits.eps
    if resolved < smallest_subnormal or resolved > limits.max:
        raise ValueError(
            "kernel_eps must remain finite and positive in the computation dtype"
        )
    return resolved


def _validate_output_dtype(output_dtype: object) -> None:
    if output_dtype is not None and (
        not isinstance(output_dtype, torch.dtype)
        or output_dtype not in _SUPPORTED_OUTPUT_DTYPES
    ):
        raise TypeError("output_dtype must be a supported floating torch.dtype")


def _infer_gmax(parameter_count: int) -> int:
    discriminant = 1 + 8 * parameter_count
    root = math.isqrt(discriminant)
    if root * root != discriminant or (root - 1) % 2 != 0:
        raise ValueError(
            "factor last dimension must be a valid packed lower-triangular size"
        )
    side = (root - 1) // 2
    gmax = side - 1
    if gmax < 1:
        raise ValueError("factor must reserve at least one dimension group")
    return gmax


def _validate_qkv(
    q: object,
    k: object,
    v: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
        raise TypeError("q, k, and v must be torch.Tensor objects")
    q_tensor, k_tensor, v_tensor = q, k, v
    if q_tensor.ndim != 4 or k_tensor.ndim != 4 or v_tensor.ndim != 4:
        raise ValueError("q, k, and v must be rank-4 head-first tensors")
    if any(
        tensor.dtype not in _SUPPORTED_FLOATING_DTYPES
        for tensor in (q_tensor, k_tensor, v_tensor)
    ):
        raise TypeError("q, k, and v must have floating-point dtypes")
    if q_tensor.dtype != k_tensor.dtype or q_tensor.dtype != v_tensor.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if q_tensor.device != k_tensor.device or q_tensor.device != v_tensor.device:
        raise ValueError("q, k, and v must be on the same device")
    if q_tensor.device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("grouped quadratic reference supports CPU, CUDA, or MPS")
    if q_tensor.shape[-1] != k_tensor.shape[-1]:
        raise ValueError("q and k must have the same feature dimension")
    if q_tensor.shape[0] != k_tensor.shape[0] or q_tensor.shape[0] != v_tensor.shape[0]:
        raise ValueError("q, k, and v must have the same batch size")
    if q_tensor.shape[2] != k_tensor.shape[2] or q_tensor.shape[2] != v_tensor.shape[2]:
        raise ValueError("q, k, and v must have the same sequence length")
    if k_tensor.shape[1] != v_tensor.shape[1]:
        raise ValueError("k and v must have the same KV head count")
    if q_tensor.shape[1] <= 0 or k_tensor.shape[1] <= 0:
        raise ValueError("q and k must have positive head counts")
    if any(dimension <= 0 for dimension in q_tensor.shape + v_tensor.shape[-1:]):
        raise ValueError("q, k, and v dimensions must be positive")
    if q_tensor.shape[1] % k_tensor.shape[1] != 0:
        raise ValueError("query head count must be divisible by KV head count")
    return q_tensor, k_tensor, v_tensor


def _validate_factor(
    factor: object,
    *,
    query_heads: int,
    q_device: torch.device,
) -> tuple[torch.Tensor, int]:
    if not isinstance(factor, torch.Tensor):
        raise TypeError("factor must be a torch.Tensor")
    if factor.ndim not in (1, 2):
        raise ValueError("factor must have shape [P] or [Hq, P]")
    if factor.dtype not in _SUPPORTED_FACTOR_DTYPES:
        raise TypeError("factor must be a floating Tensor with dtype FP32 or FP64")
    if factor.device != q_device:
        raise ValueError("factor must be on the same device as q")
    if factor.shape[-1] <= 0:
        raise ValueError("factor must contain packed lower-triangular parameters")
    gmax = _infer_gmax(factor.shape[-1])
    if factor.ndim == 2 and factor.shape[0] != query_heads:
        raise ValueError("per-query-head factor first dimension must equal Hq")
    return factor, gmax


def _canonicalize_dim_groups(
    dim_groups: object,
    *,
    query_heads: int,
    head_dimension: int,
    gmax: int,
    q_device: torch.device,
) -> torch.Tensor:
    if not isinstance(dim_groups, torch.Tensor):
        raise TypeError("dim_groups must be a torch.Tensor")
    if dim_groups.dtype not in {torch.int32, torch.int64}:
        raise TypeError("dim_groups must use torch.int32 or torch.int64")
    if dim_groups.device != q_device:
        raise ValueError("dim_groups must be on the same device as q")
    if dim_groups.ndim == 1:
        if dim_groups.shape[0] != head_dimension:
            raise ValueError("dim_groups last dimension must equal D")
        raw = dim_groups
        heads_for_validation = raw.unsqueeze(0).expand(query_heads, -1)
        canonical = raw.to(dtype=torch.int32).contiguous()
    elif dim_groups.ndim == 2:
        if dim_groups.shape[0] != query_heads:
            raise ValueError(
                "dim_groups first dimension must equal query-head count Hq"
            )
        if dim_groups.shape[1] != head_dimension:
            raise ValueError("dim_groups last dimension must equal D")
        raw = dim_groups
        heads_for_validation = raw
        canonical = raw.to(dtype=torch.int32).contiguous()
    else:
        raise ValueError("dim_groups must have shape [D] or [Hq, D]")
    if not ((heads_for_validation >= 0) & (heads_for_validation < gmax)).all():
        raise ValueError("dim_groups IDs must satisfy 0 <= id < Gmax")
    for query_head in range(query_heads):
        used = torch.unique(heads_for_validation[query_head], sorted=True)
        expected = torch.arange(
            used.numel(),
            dtype=used.dtype,
            device=used.device,
        )
        if used.numel() == 0 or not torch.equal(used, expected):
            raise ValueError(
                "dim_groups IDs for every query head must form a continuous prefix"
            )
    return canonical


def _expand_factor_coefficients(
    factor: torch.Tensor,
    *,
    query_heads: int,
    gmax: int,
    kernel_eps: float,
    working_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    factor_by_head = factor.unsqueeze(0) if factor.ndim == 1 else factor
    factor_by_head = factor_by_head.to(dtype=working_dtype)
    if factor_by_head.shape[0] == 1:
        factor_by_head = factor_by_head.expand(query_heads, -1)
    side = gmax + 1
    rows, columns = torch.tril_indices(side, side, device=factor.device)
    linear_indices = rows * side + columns
    lower_flat = torch.zeros(
        (query_heads, side * side),
        dtype=working_dtype,
        device=factor.device,
    )
    lower_flat = lower_flat.scatter(
        1,
        linear_indices.unsqueeze(0).expand(query_heads, -1),
        factor_by_head,
    )
    lower = lower_flat.reshape(query_heads, side, side)
    matrix = lower @ lower.transpose(-1, -2)
    epsilon = torch.as_tensor(kernel_eps, dtype=working_dtype, device=factor.device)
    constant = matrix[:, 0, 0] + epsilon
    linear = 2.0 * matrix[:, 0, 1:]
    # The Triton query kernel addresses C as a packed contiguous [Hq, G, G]
    # tensor. This slice is a strided view into the PSD matrix, so make the
    # layout explicit at the API/backend boundary.
    quadratic = matrix[:, 1:, 1:].contiguous()
    return constant, linear, quadratic


def _validate_public_options(
    *,
    causal: object,
    execution: object,
    output_dtype: object,
    return_unnormalized: object,
) -> tuple[bool, Literal["reference", "triton", "auto"]]:
    if not isinstance(causal, bool):
        raise TypeError("causal must be a bool")
    if not causal:
        raise ValueError(
            "grouped quadratic attention is causal-only; causal=False is no longer supported"
        )
    if not isinstance(execution, str) or execution not in {
        "reference",
        "triton",
        "auto",
    }:
        raise ValueError("execution must be one of: reference, triton, auto")
    _validate_output_dtype(output_dtype)
    if not isinstance(return_unnormalized, bool):
        raise TypeError("return_unnormalized must be a bool")
    return causal, execution  # type: ignore[return-value]


def _run_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    constant_coefficients: torch.Tensor,
    linear_coefficients: torch.Tensor,
    quadratic_coefficients: torch.Tensor,
    *,
    scale: float,
    kernel_eps: float,
    causal: bool,
    execution: Literal["triton", "auto"],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from .triton import grouped_quadratic_backward as triton_backward

    return triton_backward.grouped_quadratic_triton_forward_autograd(
        q,
        k,
        v,
        dim_groups,
        constant_coefficients,
        linear_coefficients,
        quadratic_coefficients,
        scale=scale,
        kernel_eps=kernel_eps,
        causal=causal,
        execution=execution,
    )


def oal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    factor: torch.Tensor,
    *,
    scale: float | None = None,
    kernel_eps: float = 1e-6,
    causal: bool = True,
    execution: Literal["reference", "triton", "auto"] = "auto",
    output_dtype: torch.dtype | None = None,
    return_unnormalized: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate OAL's caller-grouped full PSD quadratic attention kernel."""
    q, k, v = _validate_qkv(q, k, v)
    causal, execution = _validate_public_options(
        causal=causal,
        execution=execution,
        output_dtype=output_dtype,
        return_unnormalized=return_unnormalized,
    )
    factor, gmax = _validate_factor(
        factor,
        query_heads=q.shape[1],
        q_device=q.device,
    )
    working_dtype = _resolve_working_dtype(q.dtype, factor.dtype)
    if q.device.type == "mps" and working_dtype == torch.float64:
        raise TypeError(
            "MPS grouped quadratic reference execution does not support float64"
        )
    resolved_kernel_eps = _validate_kernel_eps(
        kernel_eps,
        working_dtype=working_dtype,
    )
    dim_groups = _canonicalize_dim_groups(
        dim_groups,
        query_heads=q.shape[1],
        head_dimension=q.shape[-1],
        gmax=gmax,
        q_device=q.device,
    )
    if scale is None:
        resolved_scale = q.shape[-1] ** -0.5
    elif isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise TypeError("scale must be a finite Python scalar")
    else:
        try:
            resolved_scale = float(scale)
        except OverflowError:
            raise ValueError("scale must be finite") from None
        if not math.isfinite(resolved_scale):
            raise ValueError("scale must be finite")

    use_reference = execution == "reference" or (
        execution == "auto" and q.device.type != "cuda"
    )
    if use_reference:
        q_work = q.to(dtype=working_dtype)
        k_work = k.to(dtype=working_dtype)
        v_work = v.to(dtype=working_dtype)
        coefficients = _expand_factor_coefficients(
            factor,
            query_heads=q.shape[1],
            gmax=gmax,
            kernel_eps=resolved_kernel_eps,
            working_dtype=working_dtype,
        )
        output, numerator, denominator = grouped_quadratic_reference(
            q_work,
            k_work,
            v_work,
            dim_groups,
            *coefficients,
            scale=resolved_scale,
            kernel_eps=resolved_kernel_eps,
            causal=causal,
        )
    else:
        if q.device.type != "cuda":
            raise RuntimeError("grouped quadratic Triton execution requires CUDA input")
        if q.dtype not in {torch.float16, torch.bfloat16}:
            raise TypeError(
                "grouped quadratic Triton execution supports only float16 and bfloat16"
            )
        if factor.dtype != torch.float32:
            raise TypeError("grouped quadratic Triton factor must use FP32")
        if q.shape[-1] > 64:
            raise ValueError("grouped quadratic Triton execution supports D<=64")
        coefficients = _expand_factor_coefficients(
            factor,
            query_heads=q.shape[1],
            gmax=gmax,
            kernel_eps=resolved_kernel_eps,
            working_dtype=torch.float32,
        )
        output, numerator, denominator = _run_triton(
            q,
            k,
            v,
            dim_groups,
            *coefficients,
            scale=resolved_scale,
            kernel_eps=resolved_kernel_eps,
            causal=causal,
            execution=execution,
        )

    if not (torch.isfinite(numerator).all() and torch.isfinite(denominator).all()):
        raise RuntimeError("Grouped Quadratic result must be finite")
    if not (denominator > 0).all():
        raise RuntimeError(
            "Grouped Quadratic denominator must be finite and strictly positive"
        )
    output = output.to(dtype=v.dtype if output_dtype is None else output_dtype)
    if not torch.isfinite(output).all():
        raise RuntimeError("Grouped Quadratic normalized output must be finite")
    if return_unnormalized:
        return output, numerator, denominator
    return output


__all__ = ["oal_attention"]
