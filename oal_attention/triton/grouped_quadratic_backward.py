"""Grouped quadratic Triton backward and its capability boundary.

The custom autograd boundary receives expanded ``A/B/C`` coefficients.  The
ordinary PyTorch factor transform therefore remains outside the custom
function and supplies the gradient back to the packed PSD factor.  The
streaming VJP produces gradients for Q/K/V and A/B/C without materializing
token-pair tensors. This module owns dispatch and the custom-autograd boundary;
reference state derivatives live in grouped_reference.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Literal

import torch

from ..grouped_observability import (
    _active_grouped_execution_trace_session,
    _observe_grouped_backward,
    _observe_grouped_stage,
    reactivate_grouped_execution_trace,
)
from . import grouped_quadratic as grouped_forward
from .grouped_quadratic_capabilities import (
    GroupedQuadraticCapability,
    GroupedQuadraticCapabilityKey,
    REVIEWED_GROUPED_QUADRATIC_CAPABILITIES,
    capability_key_is_reviewed,
    grouped_quadratic_capability_key,
)
from .grouped_quadratic_causal_common import KernelPlan
from .availability import triton_is_available
from .grouped_quadratic_production_witness import (
    active_production_witness_session,
    reactivate_production_witness_session,
)

TRITON_GROUPED_QUADRATIC_BACKWARD_VALIDATED_CAPABILITIES: frozenset[
    GroupedQuadraticCapability
] = REVIEWED_GROUPED_QUADRATIC_CAPABILITIES


def grouped_quadratic_backward_capability_is_validated(
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
    """Return true only when every real causal VJP stage has a record."""
    if (
        dim_groups is None
        or constant_coefficients is None
        or quadratic_coefficients is None
    ):
        return False
    if not causal or not q.is_cuda:
        return False
    from .grouped_quadratic_causal_backward import (
        build_causal_vjp_kernel_plans_from_metadata,
    )

    plans = build_causal_vjp_kernel_plans_from_metadata(
        q,
        k,
        v,
        dim_groups,
        constant_coefficients,
        linear_coefficients,
        quadratic_coefficients,
        needs_input_grad=(
            q.requires_grad,
            k.requires_grad,
            v.requires_grad,
            constant_coefficients.requires_grad,
            linear_coefficients.requires_grad,
            quadratic_coefficients.requires_grad,
        ),
    )
    return grouped_quadratic_backward_stage_plans_are_validated(plans)


def grouped_quadratic_backward_stage_capability_keys(
    plans: object,
) -> tuple[
    GroupedQuadraticCapabilityKey | None,
    GroupedQuadraticCapabilityKey | None,
    GroupedQuadraticCapabilityKey | None,
]:
    """Return exact reviewed-record keys for normalization, prefix, and suffix."""
    from .grouped_quadratic_causal_backward import GroupedCausalVjpKernelPlans

    if not isinstance(plans, GroupedCausalVjpKernelPlans):
        raise TypeError("backward stage keys require GroupedCausalVjpKernelPlans")
    normalization_key = (
        grouped_quadratic_capability_key(
            plans.normalization_plan,
            execution_mode="backward_normalization",
        )
        if plans.normalization_plan is not None
        else None
    )
    prefix_key = (
        grouped_quadratic_capability_key(
            plans.prefix_plan,
            execution_mode="backward_prefix",
        )
        if plans.prefix_plan is not None
        else None
    )
    suffix_key = (
        grouped_quadratic_capability_key(
            plans.suffix_plan,
            execution_mode="backward_suffix",
        )
        if plans.suffix_plan is not None
        else None
    )
    return normalization_key, prefix_key, suffix_key


def grouped_quadratic_backward_stage_plans_are_validated(plans: object) -> bool:
    """Require exact records for every VJP stage that will launch."""
    return all(
        capability_key_is_reviewed(key)
        for key in grouped_quadratic_backward_stage_capability_keys(plans)
        if key is not None
    )


class _GroupedQuadraticTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dim_groups: torch.Tensor,
        constant_coefficients: torch.Tensor,
        linear_coefficients: torch.Tensor,
        quadratic_coefficients: torch.Tensor,
        scale: float,
        causal: bool,
        kernel_plan: KernelPlan | None,
        backward_kernel_plans: object | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # PyTorch may invoke ``backward`` under a fresh ContextVar context.
        # Preserve both observation-only sinks so the VJP can reattach to the
        # same session without influencing capability admission or plan choice.
        ctx.production_witness_session = active_production_witness_session()
        ctx.grouped_execution_trace_session = _active_grouped_execution_trace_session()
        forward_scope = (
            _observe_grouped_stage("forward", kernel_plan.to_dict)
            if kernel_plan is not None
            else nullcontext()
        )
        with forward_scope:
            output, numerator, denominator = grouped_forward._launch_grouped_forward(
                q,
                k,
                v,
                dim_groups,
                constant_coefficients,
                linear_coefficients,
                quadratic_coefficients,
                scale=scale,
                causal=causal,
                kernel_plan=kernel_plan,
            )
        ctx.save_for_backward(
            q,
            k,
            v,
            dim_groups,
            constant_coefficients,
            linear_coefficients,
            quadratic_coefficients,
            numerator,
            denominator,
        )
        ctx.scale = scale
        ctx.causal = causal
        ctx.backward_kernel_plans = backward_kernel_plans
        ctx.set_materialize_grads(False)
        return output, numerator, denominator

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor | None,
        grad_numerator: torch.Tensor | None,
        grad_denominator: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        if torch.is_grad_enabled():
            raise RuntimeError(
                "grouped quadratic Triton double backward is not supported"
            )
        (
            q,
            k,
            v,
            dim_groups,
            constant_coefficients,
            linear_coefficients,
            quadratic_coefficients,
            numerator,
            denominator,
        ) = ctx.saved_tensors
        needs_input_grad = ctx.needs_input_grad
        if not any(needs_input_grad[:7]):
            return (None,) * 11
        from .grouped_quadratic_causal_backward import (
            grouped_causal_first_order_vjp,
        )

        observation_session = getattr(ctx, "production_witness_session", None)
        witness_scope = (
            reactivate_production_witness_session(observation_session)
            if observation_session is not None
            else nullcontext()
        )
        trace_session = getattr(ctx, "grouped_execution_trace_session", None)
        trace_scope = (
            reactivate_grouped_execution_trace(trace_session)
            if trace_session is not None and trace_session.is_open
            else nullcontext()
        )
        with witness_scope, trace_scope, _observe_grouped_backward():
            (
                d_q,
                d_k,
                d_v,
                d_constant,
                d_linear,
                d_quadratic,
            ) = grouped_causal_first_order_vjp(
                q,
                k,
                v,
                dim_groups,
                constant_coefficients,
                linear_coefficients,
                quadratic_coefficients,
                grad_output,
                numerator,
                denominator,
                grad_numerator=grad_numerator,
                grad_denominator=grad_denominator,
                scale=ctx.scale,
                needs_input_grad=(
                    needs_input_grad[0],
                    needs_input_grad[1],
                    needs_input_grad[2],
                    needs_input_grad[4],
                    needs_input_grad[5],
                    needs_input_grad[6],
                ),
                kernel_plans=ctx.backward_kernel_plans,
            )
        return (
            d_q,
            d_k,
            d_v,
            None,
            d_constant,
            d_linear,
            d_quadratic,
            None,
            None,
            None,
            None,
        )


def _grouped_quadratic_triton_forward_autograd_core(
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
    backward_kernel_plans: object | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared custom-autograd core for private bring-up and planned execution."""
    return _GroupedQuadraticTritonFunction.apply(
        q,
        k,
        v,
        dim_groups,
        constant_coefficients,
        linear_coefficients,
        quadratic_coefficients,
        scale,
        causal,
        kernel_plan,
        backward_kernel_plans,
    )


def grouped_quadratic_triton_forward_unvalidated_autograd(
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
    backward_kernel_plans: object | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Private bring-up entry that exercises the real Triton VJP."""
    return _grouped_quadratic_triton_forward_autograd_core(
        q,
        k,
        v,
        dim_groups,
        constant_coefficients,
        linear_coefficients,
        quadratic_coefficients,
        scale=scale,
        causal=causal,
        kernel_plan=kernel_plan,
        backward_kernel_plans=backward_kernel_plans,
    )


def grouped_quadratic_triton_backward_unvalidated_diagnostics(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    constant_coefficients: torch.Tensor,
    linear_coefficients: torch.Tensor,
    quadratic_coefficients: torch.Tensor,
    d_numerator: torch.Tensor,
    d_denominator: torch.Tensor,
    *,
    scale: float,
    causal: bool,
) -> tuple[tuple[torch.Tensor, ...], dict[str, object]]:
    """Private bring-up hook exposing backward intermediates for diagnosis."""
    if not causal:
        raise ValueError("grouped quadratic attention is causal-only")
    from .grouped_quadratic_causal_backward import (
        grouped_causal_unnormalized_vjp,
    )
    from .grouped_quadratic_causal_common import grouped_causal_state_bytes

    gradients = grouped_causal_unnormalized_vjp(
        q,
        k,
        v,
        dim_groups,
        constant_coefficients,
        linear_coefficients,
        quadratic_coefficients,
        d_numerator,
        d_denominator,
        scale=scale,
    )
    state_bytes = grouped_causal_state_bytes(
        batch_size=q.shape[0],
        key_value_heads=k.shape[1],
        head_dimension=q.shape[-1],
        value_dimension=v.shape[-1],
    )
    return gradients, {
        "scan_config": {
            "implementation": "kv_owned_streaming",
            "token_block": 1,
            "pair_block": q.shape[-1] * (q.shape[-1] + 1) // 2,
            "value_block": v.shape[-1] + 1,
            "groups_per_wave": linear_coefficients.shape[-1],
        },
        "workspace_accounting": {
            "prefix_state_bytes": state_bytes,
            "suffix_state_bytes": state_bytes,
            "block_partial_bytes": 0,
        },
    }


def grouped_quadratic_triton_forward_autograd(
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
    """Planner-gated Triton forward/backward autograd entry."""
    if not causal:
        raise ValueError("grouped quadratic attention is causal-only")
    del kernel_eps
    forward_key = grouped_forward.grouped_quadratic_capability_key_for_inputs(
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
    if forward_key is None:
        raise RuntimeError(
            "grouped quadratic Triton execution has no supported physical plan"
        )
    needs_backward = any(forward_key.requested_gradient_mask)
    backward_kernel_plans: object | None = None
    if needs_backward:
        from .grouped_quadratic_causal_backward import (
            build_causal_vjp_kernel_plans_from_metadata,
        )

        backward_kernel_plans = build_causal_vjp_kernel_plans_from_metadata(
            q,
            k,
            v,
            dim_groups,
            constant_coefficients,
            linear_coefficients,
            quadratic_coefficients,
            needs_input_grad=forward_key.requested_gradient_mask,
        )
    return grouped_quadratic_triton_forward_unvalidated_autograd(
        q,
        k,
        v,
        dim_groups,
        constant_coefficients,
        linear_coefficients,
        quadratic_coefficients,
        scale=scale,
        causal=causal,
        kernel_plan=forward_key.plan,
        backward_kernel_plans=backward_kernel_plans,
    )


def require_grouped_quadratic_backward_capability(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    linear_coefficients: torch.Tensor,
    *,
    causal: bool,
    execution: str,
) -> None:
    if q.dtype not in {torch.float16, torch.bfloat16}:
        raise TypeError(
            "grouped quadratic Triton backward supports only float16 and bfloat16"
        )
    if not 1 <= q.shape[-1] <= 64:
        raise ValueError(
            "grouped quadratic Triton backward supports head dimensions 1 through 64"
        )
    if not grouped_quadratic_backward_capability_is_validated(
        q,
        k,
        v,
        linear_coefficients,
        causal=causal,
    ):
        raise RuntimeError(
            "strict grouped quadratic backward evidence check requires a current "
            "exact reviewed record for the planned physical path"
        )
