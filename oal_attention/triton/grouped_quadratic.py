"""Grouped quadratic Triton forward path and its capability boundary.

The grouped kernels use KV-owned H0/H1/H2 states.  H2 is stored in packed
lower-triangular order, so the production state size is proportional to
``D * (D + 1) // 2`` rather than an ordered ``D * D`` pair space.  Capability
records describe profiles that underwent separate source and hardware review;
runtime support is determined independently by the live planner.
"""

from __future__ import annotations

from typing import Literal

import torch

from .grouped_quadratic_capabilities import (
    GroupedQuadraticCapability,
    GroupedQuadraticCapabilityKey,
    REVIEWED_GROUPED_QUADRATIC_CAPABILITIES,
    capability_key_is_reviewed,
    grouped_quadratic_capability_key,
)
from .grouped_quadratic_causal_common import KernelPlan
from .availability import triton, triton_is_available

TRITON_GROUPED_QUADRATIC_VALIDATED_CAPABILITIES: frozenset[
    GroupedQuadraticCapability
] = REVIEWED_GROUPED_QUADRATIC_CAPABILITIES


def grouped_quadratic_capability_key_for_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    constant_coefficients: torch.Tensor,
    linear_coefficients: torch.Tensor,
    quadratic_coefficients: torch.Tensor,
    *,
    causal: bool,
    execution_mode: Literal["forward"],
) -> GroupedQuadraticCapabilityKey | None:
    """Build the live causal plan key before examining reviewed records.

    Public dispatch always derives a fresh plan from the actual CUDA geometry,
    workspace budget, tensor layout, and Q/K/V/A/B/C gradient mask.  It never
    accepts an externally supplied plan identity.
    """
    if not causal:
        return None
    if not q.is_cuda:
        return None
    from .grouped_quadratic_causal_forward_kernels import build_forward_kernel_plan

    kernel_plan = build_forward_kernel_plan(
        q,
        k,
        v,
        dim_groups,
        constant_coefficients,
        linear_coefficients,
        quadratic_coefficients,
    )
    return grouped_quadratic_capability_key(
        kernel_plan,
        execution_mode=execution_mode,
    )


def grouped_quadratic_capability_is_validated(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    linear_coefficients: torch.Tensor,
    *,
    causal: bool,
    dim_groups: torch.Tensor | None = None,
    constant_coefficients: torch.Tensor | None = None,
    quadratic_coefficients: torch.Tensor | None = None,
) -> bool:
    """Return true only when a complete live forward key has a record."""
    # The optional arguments preserve the Task-0 diagnostic predicate for
    # callers that only hold the retired nine-field geometry.  That incomplete
    # input can never produce an admission key.
    if (
        dim_groups is None
        or constant_coefficients is None
        or quadratic_coefficients is None
    ):
        return False
    key = grouped_quadratic_capability_key_for_inputs(
        q,
        k,
        v,
        dim_groups,
        constant_coefficients,
        linear_coefficients,
        quadratic_coefficients,
        causal=causal,
        execution_mode="forward",
    )
    return key is not None and capability_key_is_reviewed(key)


def _require_forward_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    constant_coefficients: torch.Tensor,
    linear_coefficients: torch.Tensor,
    quadratic_coefficients: torch.Tensor,
) -> None:
    if not triton_is_available():
        raise RuntimeError(
            "grouped quadratic Triton execution requires Triton with triton.language"
        )
    if not all(tensor.is_cuda for tensor in (q, k, v)):
        raise RuntimeError("grouped quadratic Triton execution requires CUDA tensors")
    if q.dtype not in {torch.float16, torch.bfloat16}:
        raise TypeError(
            "grouped quadratic Triton execution supports only float16 and bfloat16"
        )
    if q.shape[-1] > 64:
        raise ValueError("grouped quadratic Triton execution supports D<=64")
    if q.shape[-1] < 1:
        raise ValueError("grouped quadratic Triton execution requires D>=1")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k, and v must have one common dtype")
    if dim_groups.dtype != torch.int32 or not dim_groups.is_contiguous():
        raise TypeError("canonical dim_groups must be contiguous int32")
    if constant_coefficients.dtype != torch.float32:
        raise TypeError("Triton grouped quadratic coefficients must be FP32")
    if linear_coefficients.dtype != torch.float32:
        raise TypeError("Triton grouped quadratic coefficients must be FP32")
    if quadratic_coefficients.dtype != torch.float32:
        raise TypeError("Triton grouped quadratic coefficients must be FP32")
    devices = {
        q.device,
        k.device,
        v.device,
        dim_groups.device,
        constant_coefficients.device,
        linear_coefficients.device,
        quadratic_coefficients.device,
    }
    if len(devices) != 1:
        raise ValueError("grouped quadratic Triton tensors must share one device")


def _launch_grouped_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    constant_coefficients: torch.Tensor,
    linear_coefficients: torch.Tensor,
    quadratic_coefficients: torch.Tensor,
    *,
    scale: float,
    causal: bool,
    kernel_plan: KernelPlan | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not causal:
        raise ValueError("grouped quadratic attention is causal-only")
    _require_forward_inputs(
        q,
        k,
        v,
        dim_groups,
        constant_coefficients,
        linear_coefficients,
        quadratic_coefficients,
    )
    assert triton is not None
    if kernel_plan is not None:
        from .grouped_quadratic_causal_forward_kernels import (
            grouped_causal_triton_forward,
        )

        return grouped_causal_triton_forward(
            q,
            k,
            v,
            dim_groups,
            constant_coefficients,
            linear_coefficients,
            quadratic_coefficients,
            scale=scale,
            kernel_plan=kernel_plan,
        )
    from .grouped_quadratic_causal_forward import (
        grouped_causal_streaming_forward,
    )

    return grouped_causal_streaming_forward(
        q,
        k,
        v,
        dim_groups,
        constant_coefficients,
        linear_coefficients,
        quadratic_coefficients,
        scale=scale,
    )


def grouped_quadratic_triton_forward(
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
    """Launch grouped forward for a live planner-supported physical path."""
    if not causal:
        raise ValueError("grouped quadratic attention is causal-only")
    del kernel_eps
    capability_key = grouped_quadratic_capability_key_for_inputs(
        q,
        k,
        v,
        dim_groups,
        constant_coefficients,
        linear_coefficients,
        quadratic_coefficients,
        causal=causal,
        execution_mode="forward",
    )
    if capability_key is None:
        raise RuntimeError(
            "grouped quadratic Triton execution has no supported physical plan"
        )
    return _launch_grouped_forward(
        q,
        k,
        v,
        dim_groups,
        constant_coefficients,
        linear_coefficients,
        quadratic_coefficients,
        scale=scale,
        causal=causal,
        kernel_plan=capability_key.plan,
    )


def grouped_quadratic_triton_forward_unvalidated(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    constant_coefficients: torch.Tensor,
    linear_coefficients: torch.Tensor,
    quadratic_coefficients: torch.Tensor,
    *,
    scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Private CUDA bring-up hook; never used by public dispatch."""
    return _launch_grouped_forward(
        q,
        k,
        v,
        dim_groups,
        constant_coefficients,
        linear_coefficients,
        quadratic_coefficients,
        scale=scale,
        causal=causal,
    )
