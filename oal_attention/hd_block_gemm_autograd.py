"""Private first-order autograd boundary for planned HD Block-GEMM."""

from __future__ import annotations

from typing import Any

import torch

from .hd_block_gemm import HDParallelBlockPlan, canonical_pair_layout
from .hd_block_gemm_backward import (
    _augmented_normalization_vjp,
    _exclusive_right_scan,
    _kv_side_vjp,
    _query_dcarry,
    _query_side_vjp,
    _shared_wave_vjp,
    _stage_backward_gradient,
)
from .hd_block_gemm_cache import CanonicalPairLayout
from .hd_block_gemm_feature_context import _prepare_feature_context
from .hd_block_gemm_profiling import (
    _capture_hd_stage_profiling_state,
    _resume_hd_stage_profiling_state,
)
from .hd_block_gemm_runtime import _saved_hd_parallel_block_gemm

_INPUT_NAMES = ("q", "k", "v", "a", "b", "c")


def _required_saved_inputs(
    gradient_mask: tuple[bool, bool, bool, bool, bool, bool],
) -> tuple[str, ...]:
    need_q, need_k, need_v, need_a, need_b, need_c = gradient_mask
    query_side = need_q or need_a or need_b or need_c
    right_scan = need_k or need_v
    names: list[str] = []
    if need_q or right_scan or need_b or need_c:
        names.append("q")
    if any(gradient_mask):
        names.append("k")
    if query_side or need_k:
        names.append("v")
    if right_scan:
        names.append("a")
    if need_q or right_scan:
        names.extend(("b", "c"))
    return tuple(names)


def _require_live_metadata(ctx: Any) -> tuple[HDParallelBlockPlan, CanonicalPairLayout]:
    plan = ctx.plan
    layout = ctx.layout
    if not isinstance(plan, HDParallelBlockPlan) or id(plan) != ctx.plan_object_id:
        raise RuntimeError("saved HD Block-GEMM plan identity changed before backward")
    if plan.plan_id != ctx.plan_id:
        raise RuntimeError("saved HD Block-GEMM plan id changed before backward")
    if (
        not isinstance(layout, CanonicalPairLayout)
        or id(layout) != ctx.layout_object_id
        or layout.layout_id != ctx.layout_id
        or layout.layout_id != plan.pair_layout_id
        or layout != canonical_pair_layout(plan.head_dimension)
    ):
        raise RuntimeError(
            "saved HD Block-GEMM layout identity changed before backward"
        )
    if plan.requested_gradient_mask != ctx.gradient_mask:
        raise RuntimeError("saved HD Block-GEMM gradient mask changed before backward")
    return plan, layout


def _cast_requested_gradient(
    gradient: torch.Tensor | None,
    *,
    requested: bool,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if not requested:
        return None
    if gradient is None:
        raise RuntimeError("requested HD Block-GEMM gradient was not produced")
    return gradient.to(dtype=dtype)


class _HDParallelBlockGemmFunction(torch.autograd.Function):
    """Connect the one-pass planned forward to its analytic first-order VJP."""

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        scale: float,
        eps: float,
        layout: CanonicalPairLayout,
        plan: HDParallelBlockPlan,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gradient_mask = tuple(bool(value) for value in ctx.needs_input_grad[:6])
        if plan.requested_gradient_mask != gradient_mask:
            raise ValueError(
                "plan gradient mask must exactly match the six tensor inputs"
            )
        ctx.set_materialize_grads(False)
        (
            output,
            numerator,
            denominator,
            carry,
            contraction_backend_token,
            local_score_tc,
            phi_k_tc,
        ) = _saved_hd_parallel_block_gemm(
            q,
            k,
            v,
            a,
            b,
            c,
            scale=scale,
            eps=eps,
            layout=layout,
            plan=plan,
        )

        ctx.hd_profile_state = _capture_hd_stage_profiling_state()
        ctx.scale = float(scale)
        ctx.eps = float(eps)
        ctx.layout = layout
        ctx.layout_id = layout.layout_id
        ctx.layout_object_id = id(layout)
        ctx.plan = plan
        ctx.plan_id = plan.plan_id
        ctx.plan_object_id = id(plan)
        ctx.gradient_mask = gradient_mask
        ctx.input_dtypes = tuple(tensor.dtype for tensor in (q, k, v, a, b, c))
        ctx.forward_contraction_backend_token = contraction_backend_token
        ctx.forward_contraction_backend_token_object_id = id(contraction_backend_token)
        ctx.forward_contraction_backend_load_generation = (
            contraction_backend_token.load_generation
        )

        input_values = dict(zip(_INPUT_NAMES, (q, k, v, a, b, c), strict=True))
        saved_names = list(_required_saved_inputs(gradient_mask))
        saved_tensors = [input_values[name] for name in saved_names]
        if any(gradient_mask):
            saved_names.extend(("numerator", "denominator"))
            saved_tensors.extend((numerator, denominator))
        query_side = gradient_mask[0] or any(gradient_mask[3:])
        if query_side:
            if carry is None:
                raise RuntimeError("query-side backward requires saved forward carry")
            saved_names.append("carry")
            saved_tensors.append(carry)
        elif carry is not None:
            raise RuntimeError("KV-only backward must not retain forward carry")
        if plan.save_local_score:
            if local_score_tc is None:
                raise RuntimeError("saved local score plan did not retain its cache")
            saved_names.append("local_score_tc")
            saved_tensors.append(local_score_tc)
        elif local_score_tc is not None:
            raise RuntimeError("recompute plan unexpectedly retained a local score")
        if plan.key_retention == "backward":
            if phi_k_tc is None:
                raise RuntimeError("backward key retention did not retain its cache")
            saved_names.append("phi_k_tc")
            saved_tensors.append(phi_k_tc)
        elif phi_k_tc is not None:
            raise RuntimeError("non-backward key retention retained a PhiK cache")
        ctx.saved_names = tuple(saved_names)
        ctx.save_for_backward(*saved_tensors)
        return output, numerator, denominator

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor | None,
        grad_numerator: torch.Tensor | None,
        grad_denominator: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
    ]:
        if torch.is_grad_enabled():
            raise RuntimeError("HD Block-GEMM does not support double backward")
        with _resume_hd_stage_profiling_state(getattr(ctx, "hd_profile_state", None)):
            return _hd_parallel_block_gemm_autograd_backward_impl(
                ctx,
                grad_output,
                grad_numerator,
                grad_denominator,
            )


def _hd_parallel_block_gemm_autograd_backward_impl(
    ctx: Any,
    grad_output: torch.Tensor | None,
    grad_numerator: torch.Tensor | None,
    grad_denominator: torch.Tensor | None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    None,
    None,
    None,
    None,
]:
    """Run analytic VJP after restoring the forward profile state, if any."""
    plan, layout = _require_live_metadata(ctx)
    saved = dict(zip(ctx.saved_names, ctx.saved_tensors, strict=True))
    if "numerator" not in saved or "denominator" not in saved:
        raise RuntimeError("HD Block-GEMM backward is missing normalization state")

    need_q, need_k, need_v, need_a, need_b, need_c = ctx.gradient_mask
    prepared = _prepare_feature_context(
        saved.get("a"),
        saved.get("b"),
        saved.get("c"),
        scale=ctx.scale,
        layout=layout,
        plan=plan,
        device=torch.device(plan.device),
        mode="backward",
    )
    with torch.no_grad(), prepared:
        backward_token = prepared.contraction_backend_token
        if (
            backward_token is None
            or backward_token is not ctx.forward_contraction_backend_token
            or id(backward_token) != ctx.forward_contraction_backend_token_object_id
            or backward_token.load_generation
            != ctx.forward_contraction_backend_load_generation
        ):
            raise RuntimeError(
                "loaded contraction backend token changed before backward"
            )
        augmented_gradient = _augmented_normalization_vjp(
            grad_output,
            grad_numerator,
            grad_denominator,
            numerator=saved["numerator"],
            denominator=saved["denominator"],
            eps=ctx.eps,
            prepared=prepared,
        )
        full_bf16_gradient = _stage_backward_gradient(
            augmented_gradient,
            prepared=prepared,
        )
        query_side = need_q or need_a or need_b or need_c
        if plan.backward_schedule == "shared_wave":
            dcarry = _query_dcarry(
                saved["q"],
                augmented_gradient,
                full_bf16_gradient=full_bf16_gradient,
                prepared=prepared,
            )
            dt = _exclusive_right_scan(dcarry, prepared=prepared)
            del dcarry
            (
                grad_q,
                grad_k,
                grad_v,
                grad_a,
                grad_b,
                grad_c,
            ) = _shared_wave_vjp(
                saved["q"],
                saved["k"],
                saved["v"],
                augmented_gradient,
                saved["carry"],
                dt,
                full_bf16_gradient=full_bf16_gradient,
                saved_local_score_tc=saved.get("local_score_tc"),
                saved_phi_k_tc=saved.get("phi_k_tc"),
                prepared=prepared,
            )
        else:
            if query_side:
                grad_q, grad_a, grad_b, grad_c = _query_side_vjp(
                    saved.get("q"),
                    saved["k"],
                    saved["v"],
                    augmented_gradient,
                    saved["carry"],
                    full_bf16_gradient=full_bf16_gradient,
                    saved_phi_k_tc=saved.get("phi_k_tc"),
                    prepared=prepared,
                )
            else:
                grad_q = grad_a = grad_b = grad_c = None

            if need_k or need_v:
                dcarry = _query_dcarry(
                    saved["q"],
                    augmented_gradient,
                    full_bf16_gradient=full_bf16_gradient,
                    prepared=prepared,
                )
                dt = _exclusive_right_scan(dcarry, prepared=prepared)
                del dcarry
                grad_k, grad_v = _kv_side_vjp(
                    saved["q"],
                    saved["k"],
                    saved.get("v"),
                    augmented_gradient,
                    dt,
                    full_bf16_gradient=full_bf16_gradient,
                    saved_local_score_tc=saved.get("local_score_tc"),
                    saved_phi_k_tc=saved.get("phi_k_tc"),
                    prepared=prepared,
                )
            else:
                grad_k = grad_v = None

    gradients = (grad_q, grad_k, grad_v, grad_a, grad_b, grad_c)
    cast_gradients = tuple(
        _cast_requested_gradient(
            gradient,
            requested=requested,
            dtype=dtype,
        )
        for gradient, requested, dtype in zip(
            gradients,
            ctx.gradient_mask,
            ctx.input_dtypes,
            strict=True,
        )
    )
    return (*cast_gradients, None, None, None, None)


def _hd_parallel_block_gemm_autograd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    scale: float,
    eps: float,
    layout: CanonicalPairLayout,
    plan: HDParallelBlockPlan,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _HDParallelBlockGemmFunction.apply(
        q,
        k,
        v,
        a,
        b,
        c,
        scale,
        eps,
        layout,
        plan,
    )


__all__: tuple[str, ...] = ()
