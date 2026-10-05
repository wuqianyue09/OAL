"""Private forward-only causal runtime for planned HD Block-GEMM.

The implementation follows three deliberately separate phases: every key
block total is written, one full-sequence exclusive scan converts totals to
carry, and only then are query/local contractions allowed to run.  All large
storage is owned by a canonical :class:`HDParallelBlockPlan` buffer.
"""

from __future__ import annotations

import math
from numbers import Real

import torch

from . import hd_block_gemm_normalization_ops
from .hd_block_gemm import (
    HDParallelBlockPlan,
    _feature_storage_for_precision,
    _require_precision_policy,
    canonical_pair_layout,
)
from .hd_block_gemm_cache import CanonicalPairLayout
from .hd_block_gemm_feature_context import (
    _PreparedFeatureContext,
    _planned_empty,
    _prepare_feature_context,
    _require_prepared,
)
from .hd_block_gemm_feature_forward import (
    _build_key_features,
    _build_query_features,
)
from .hd_block_gemm_contractions import _bmm_fp32
from .hd_cublas_compat import LoadedHdContractionBackendToken
from .hd_block_gemm_profiling import (
    _record_hd_bmm_stage,
    _record_hd_contraction,
    _record_hd_stage,
)

_PHYSICAL_PATH = "causal_parallel_block_gemm"
_FORWARD_ONLY = (False, False, False, False, False, False)
_FP32_MAX = torch.finfo(torch.float32).max
_FP32_HALF_MIN_SUBNORMAL = math.ldexp(1.0, -150)
_DENOMINATOR_ERROR = "denominator must be finite and strictly positive"


def _cumsum(
    source: torch.Tensor,
    *,
    dim: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Patchable seam that still observes the live torch cumsum operator."""
    return torch.cumsum(source, dim=dim, out=out)


def _require_forward_only_no_grad(*values: object) -> None:
    """Reject live autograd before cache admission or planned allocation."""
    if torch.is_grad_enabled() and any(
        isinstance(value, torch.Tensor) and value.requires_grad for value in values
    ):
        raise RuntimeError(
            "HD Block-GEMM forward-only runtime rejects requires_grad tensors "
            "while gradient recording is enabled"
        )


def _require_eps(eps: object) -> float:
    if not isinstance(eps, Real) or isinstance(eps, bool):
        raise TypeError("eps must be a real number")
    try:
        numeric = float(eps)
    except (OverflowError, ValueError) as error:
        raise ValueError(
            "eps must be finite, positive, and representable in FP32"
        ) from error
    if (
        not math.isfinite(numeric)
        or numeric <= _FP32_HALF_MIN_SUBNORMAL
        or numeric > _FP32_MAX
    ):
        raise ValueError("eps must be finite, positive, and representable in FP32")
    return numeric


def _require_runtime_plan(
    q: object,
    k: object,
    v: object,
    *,
    layout: object,
    plan: object,
    require_forward_only: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    CanonicalPairLayout,
    HDParallelBlockPlan,
]:
    if not isinstance(plan, HDParallelBlockPlan):
        raise TypeError("plan must be an HDParallelBlockPlan")
    if plan.physical_path != _PHYSICAL_PATH:
        raise ValueError("plan physical path must be causal_parallel_block_gemm")
    try:
        expected_feature_storage = _feature_storage_for_precision(plan.precision)
    except ValueError as error:
        raise ValueError("plan precision is unsupported") from error
    if plan.feature_storage != expected_feature_storage:
        raise ValueError("plan feature storage does not match precision")
    if require_forward_only and plan.requested_gradient_mask != _FORWARD_ONLY:
        raise ValueError("HD Block-GEMM runtime requires a forward-only plan")
    _require_precision_policy(
        plan.device_type,
        plan.precision,
        input_dtype=plan.input_dtype,
    )
    if not all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
        raise TypeError("Q, K, and V must be tensors")
    q_tensor, k_tensor, v_tensor = q, k, v
    expected_dtype = getattr(torch, plan.input_dtype)
    expected = (
        (
            q_tensor,
            (
                plan.batch_size,
                plan.query_heads,
                plan.sequence_length,
                plan.head_dimension,
            ),
            plan.input_strides[0],
        ),
        (
            k_tensor,
            (
                plan.batch_size,
                plan.key_value_heads,
                plan.sequence_length,
                plan.head_dimension,
            ),
            plan.input_strides[1],
        ),
        (
            v_tensor,
            (
                plan.batch_size,
                plan.key_value_heads,
                plan.sequence_length,
                plan.value_dimension,
            ),
            plan.input_strides[2],
        ),
    )
    if any(
        tensor.layout != torch.strided
        or tuple(tensor.shape) != shape
        or tuple(tensor.stride()) != strides
        or tensor.dtype != expected_dtype
        or str(tensor.device) != plan.device
        for tensor, shape, strides in expected
    ):
        raise ValueError("Q, K, and V do not match the plan")
    if not isinstance(layout, CanonicalPairLayout):
        raise TypeError("layout must be a CanonicalPairLayout")
    if layout != canonical_pair_layout(plan.head_dimension):
        raise ValueError("layout must be canonical for the plan head dimension")
    if layout.layout_id != plan.pair_layout_id:
        raise ValueError("layout does not match the plan")
    return q_tensor, k_tensor, v_tensor, layout, plan


def _allocate_planned(
    prepared: _PreparedFeatureContext,
    *,
    name: str,
) -> torch.Tensor:
    """Allocate exactly one named plan buffer under the invocation lease."""
    prepared = _require_prepared(prepared, mode="forward")
    return _planned_empty(prepared, name)


def _prepare_key_value_wave(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    prepared: _PreparedFeatureContext,
    u_wave: torch.Tensor,
    block_start: int,
    wave_blocks: int,
    plan: HDParallelBlockPlan,
    phi_k_cache_wave: torch.Tensor | None = None,
    reuse_key_features: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    token_start = block_start * plan.token_block
    valid_wave_tokens = wave_blocks * plan.token_block
    capacity_tokens = plan.feature_wave_blocks * plan.token_block
    token_end = min(plan.sequence_length, token_start + valid_wave_tokens)
    valid_tokens = token_end - token_start
    if phi_k_cache_wave is None:
        if reuse_key_features:
            raise ValueError("key features cannot be reused without a cache wave")
        phi_k_valid = _build_key_features(
            k[:, :, token_start:token_end, :], prepared=prepared
        )
        workspace = prepared.forward_workspace
        if workspace is None or workspace.phi_k_output is None:
            raise RuntimeError("forward key workspace was not initialized")
        phi_k_flat = workspace.phi_k_output[:, :, :capacity_tokens, :]
    else:
        phi_k_flat = phi_k_cache_wave.view(
            plan.batch_size,
            plan.key_value_heads,
            capacity_tokens,
            plan.physical_feature_dimension,
        )
        if reuse_key_features:
            phi_k_valid = phi_k_flat[:, :, :valid_tokens, :]
        else:
            phi_k_valid = _build_key_features(
                k[:, :, token_start:token_end, :],
                prepared=prepared,
                out=phi_k_flat[:, :, :valid_tokens, :],
            )
    expected_phi_k_valid = phi_k_flat[:, :, :valid_tokens, :]
    strides_match = all(
        dimension == 1 or actual_stride == expected_stride
        for dimension, actual_stride, expected_stride in zip(
            phi_k_valid.shape,
            phi_k_valid.stride(),
            expected_phi_k_valid.stride(),
            strict=True,
        )
    )
    if (
        tuple(phi_k_valid.shape) != tuple(expected_phi_k_valid.shape)
        or not strides_match
        or phi_k_valid.dtype != expected_phi_k_valid.dtype
        or phi_k_valid.device != expected_phi_k_valid.device
        or phi_k_valid.storage_offset() != expected_phi_k_valid.storage_offset()
        or phi_k_valid.untyped_storage().data_ptr()
        != expected_phi_k_valid.untyped_storage().data_ptr()
    ):
        raise RuntimeError("key feature builder returned an invalid wave view")
    if not reuse_key_features and valid_tokens < capacity_tokens:
        phi_k_flat[:, :, valid_tokens:capacity_tokens, :].zero_()
    u_wave[:, :, :valid_tokens, : plan.value_dimension].copy_(
        v[:, :, token_start:token_end, :]
    )
    u_wave[:, :, :valid_tokens, plan.value_dimension].fill_(1.0)
    if valid_tokens < capacity_tokens:
        u_wave[:, :, valid_tokens:capacity_tokens, :].zero_()
    phi_k = phi_k_flat.view(
        plan.batch_size,
        plan.key_value_heads,
        plan.feature_wave_blocks,
        plan.token_block,
        plan.physical_feature_dimension,
    )
    u = u_wave.view(
        plan.batch_size,
        plan.key_value_heads,
        plan.feature_wave_blocks,
        plan.token_block,
        plan.augmented_value_dimension,
    )
    return phi_k, u, valid_tokens


def _write_block_totals_wave(
    phi_k: torch.Tensor,
    *,
    prepared: _PreparedFeatureContext,
    u_wave: torch.Tensor,
    block_totals: torch.Tensor,
    block_totals_wave: torch.Tensor,
    block_start: int,
    wave_blocks: int,
    valid_tokens: int,
    plan: HDParallelBlockPlan,
) -> None:
    del valid_tokens
    flat_batches = plan.batch_size * plan.key_value_heads * plan.feature_wave_blocks
    phi_k_3d = phi_k.view(
        flat_batches,
        plan.token_block,
        plan.physical_feature_dimension,
    )
    u_3d = u_wave.view(
        flat_batches,
        plan.token_block,
        plan.augmented_value_dimension,
    )
    totals_3d = block_totals_wave.view(
        flat_batches,
        plan.physical_feature_dimension,
        plan.augmented_value_dimension,
    )
    _bmm_fp32(
        phi_k_3d.transpose(1, 2),
        u_3d,
        out=totals_3d,
        precision=plan.precision,
        backend_identity=plan.contraction_backend_identity,
        loaded_backend_token=prepared.contraction_backend_token,
    )
    block_totals[:, :, block_start : block_start + wave_blocks, :, :].copy_(
        block_totals_wave[:, :, :wave_blocks, :, :]
    )


def _exclusive_cumsum_block_totals(
    block_totals: torch.Tensor,
    scan_inclusive: torch.Tensor,
) -> None:
    """Convert totals to carry with one cumsum over the complete NB axis."""
    _cumsum(block_totals, dim=2, out=scan_inclusive)
    block_totals[:, :, 1:, :, :].copy_(scan_inclusive[:, :, :-1, :, :])
    block_totals[:, :, 0, :, :].zero_()


def _prepare_query_wave(
    q: torch.Tensor,
    *,
    prepared: _PreparedFeatureContext,
    block_start: int,
    wave_blocks: int,
    valid_tokens: int,
    plan: HDParallelBlockPlan,
) -> torch.Tensor:
    token_start = block_start * plan.token_block
    token_end = token_start + valid_tokens
    _build_query_features(q[:, :, token_start:token_end, :], prepared=prepared)
    workspace = prepared.forward_workspace
    if workspace is None:
        raise RuntimeError("forward feature workspace was not initialized")
    capacity_tokens = plan.feature_wave_blocks * plan.token_block
    phi_q_flat = workspace.phi_q_output[:, :, :capacity_tokens, :]
    if valid_tokens < capacity_tokens:
        phi_q_flat[:, :, valid_tokens:capacity_tokens, :].zero_()
    return phi_q_flat.view(
        plan.batch_size,
        plan.query_heads,
        plan.feature_wave_blocks,
        plan.token_block,
        plan.physical_feature_dimension,
    )


def _write_and_normalize_query_wave(
    y_wave: torch.Tensor,
    output: torch.Tensor,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    *,
    block_start: int,
    valid_tokens: int,
    eps: float,
    plan: HDParallelBlockPlan,
) -> None:
    if plan.forward_normalize_impl == "triton":
        hd_block_gemm_normalization_ops.forward_normalize_triton(
            y_wave,
            output,
            numerator,
            denominator,
            block_start=block_start,
            valid_tokens=valid_tokens,
            eps=eps,
            plan=plan,
        )
        return
    token_start = block_start * plan.token_block
    token_end = token_start + valid_tokens
    y_valid = y_wave[:, :, :valid_tokens, :]
    numerator[:, :, token_start:token_end, :].copy_(
        y_valid[..., : plan.value_dimension]
    )
    denominator[:, :, token_start:token_end, :].copy_(
        y_valid[..., plan.value_dimension :]
    )
    y_valid[..., plan.value_dimension :].clamp_min_(eps)
    y_valid[..., : plan.value_dimension].div_(y_valid[..., plan.value_dimension :])
    output[:, :, token_start:token_end, :].copy_(y_valid[..., : plan.value_dimension])


def _scalar_to_float(value: torch.Tensor) -> float:
    return float(value.item())


def _require_positive_denominator(denominator: torch.Tensor) -> None:
    """Validate the complete written denominator with one scalar reduction."""
    minimum, maximum = torch.aminmax(denominator)
    # These two host synchronizations are intentional and are a known forward
    # performance cost.  A device-side assertion can poison a CUDA context, so
    # ordinary invalid-input handling must remain synchronously recoverable
    # until a safe asynchronous status mechanism is available and benchmarked.
    minimum_value = _scalar_to_float(minimum)
    maximum_value = _scalar_to_float(maximum)
    if (
        not math.isfinite(minimum_value)
        or not math.isfinite(maximum_value)
        or minimum_value <= 0.0
    ):
        raise RuntimeError(
            f"{_DENOMINATOR_ERROR} (min={minimum_value}, max={maximum_value})"
        )


def _contract_query_wave(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    prepared: _PreparedFeatureContext,
    carry: torch.Tensor,
    u_wave: torch.Tensor,
    phi_k_query_wave: torch.Tensor | None,
    carry_query_wave: torch.Tensor,
    u_query_wave: torch.Tensor | None,
    local_score: torch.Tensor,
    local_score_tc: torch.Tensor | None,
    local_y: torch.Tensor | None,
    y_wave: torch.Tensor,
    output: torch.Tensor,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    block_start: int,
    wave_blocks: int,
    eps: float,
    plan: HDParallelBlockPlan,
    phi_k_cache_wave: torch.Tensor | None = None,
) -> None:
    with _record_hd_stage("hd.forward.k_feature_local"):
        phi_k, u, valid_tokens = _prepare_key_value_wave(
            k,
            v,
            prepared=prepared,
            u_wave=u_wave,
            block_start=block_start,
            wave_blocks=wave_blocks,
            plan=plan,
            phi_k_cache_wave=phi_k_cache_wave,
            reuse_key_features=phi_k_cache_wave is not None,
        )
    with _record_hd_stage("hd.forward.q_feature"):
        phi_q = _prepare_query_wave(
            q,
            prepared=prepared,
            block_start=block_start,
            wave_blocks=wave_blocks,
            valid_tokens=valid_tokens,
            plan=plan,
        )
    phi_k_for_query = phi_k
    u_for_query = u
    if plan.gqa_ratio > 1:
        if phi_k_query_wave is None or u_query_wave is None:
            raise RuntimeError("planned GQA query-wave storage was not allocated")
        phi_k_for_query = phi_k_query_wave.view(
            plan.batch_size,
            plan.query_heads,
            plan.feature_wave_blocks,
            plan.token_block,
            plan.physical_feature_dimension,
        )
        u_for_query = u_query_wave.view(
            plan.batch_size,
            plan.query_heads,
            plan.feature_wave_blocks,
            plan.token_block,
            plan.augmented_value_dimension,
        )
        with _record_hd_stage("hd.forward.gqa_phi_k_u_copy"):
            phi_k_for_query.view(
                plan.batch_size,
                plan.key_value_heads,
                plan.gqa_ratio,
                plan.feature_wave_blocks,
                plan.token_block,
                plan.physical_feature_dimension,
            ).copy_(phi_k.unsqueeze(2))
            u_for_query.view(
                plan.batch_size,
                plan.key_value_heads,
                plan.gqa_ratio,
                plan.feature_wave_blocks,
                plan.token_block,
                plan.augmented_value_dimension,
            ).copy_(u.unsqueeze(2))

    carry_query_blocks = carry_query_wave.view(
        plan.batch_size,
        plan.key_value_heads,
        plan.gqa_ratio,
        plan.feature_wave_blocks,
        plan.physical_feature_dimension,
        plan.augmented_value_dimension,
    )
    with _record_hd_stage("hd.forward.gqa_carry_copy"):
        if wave_blocks < plan.feature_wave_blocks:
            carry_query_blocks[:, :, :, wave_blocks:, :, :].zero_()
        carry_query_blocks[:, :, :, :wave_blocks, :, :].copy_(
            carry[
                :,
                :,
                block_start : block_start + wave_blocks,
                :,
                :,
            ].unsqueeze(2)
        )

    flat_batches = plan.batch_size * plan.query_heads * plan.feature_wave_blocks
    query_3d = phi_q.view(
        flat_batches,
        plan.token_block,
        plan.physical_feature_dimension,
    )
    key_3d = phi_k_for_query.view(
        flat_batches,
        plan.token_block,
        plan.physical_feature_dimension,
    )
    score_3d = local_score.view(
        flat_batches,
        plan.token_block,
        plan.token_block,
    )
    with _record_hd_stage("hd.forward.local_score"):
        _bmm_fp32(
            query_3d,
            key_3d.transpose(1, 2),
            out=score_3d,
            precision=plan.precision,
            backend_identity=plan.contraction_backend_identity,
            loaded_backend_token=prepared.contraction_backend_token,
        )
    with _record_hd_stage("hd.forward.local_mask"):
        score_3d.tril_()

    y_3d = y_wave.view(
        flat_batches,
        plan.token_block,
        plan.augmented_value_dimension,
    )
    carry_3d = carry_query_wave.view(
        flat_batches,
        plan.physical_feature_dimension,
        plan.augmented_value_dimension,
    )
    with _record_hd_stage("hd.forward.cross"):
        _bmm_fp32(
            query_3d,
            carry_3d,
            out=y_3d,
            precision=plan.precision,
            backend_identity=plan.contraction_backend_identity,
            loaded_backend_token=prepared.contraction_backend_token,
        )
    u_3d = u_for_query.view(
        flat_batches,
        plan.token_block,
        plan.augmented_value_dimension,
    )
    if plan.precision == "bf16_tensorcore":
        if local_score_tc is None or local_y is None:
            raise RuntimeError(
                "Tensor-Core local contraction scratch was not allocated"
            )
        score_tc_3d = local_score_tc.view(
            flat_batches,
            plan.token_block,
            plan.token_block,
        )
        with _record_hd_stage("hd.forward.local_score_cast"):
            score_tc_3d.copy_(score_3d)
        local_y_3d = local_y.view(
            flat_batches,
            plan.token_block,
            plan.augmented_value_dimension,
        )
        with _record_hd_stage("hd.forward.local_score_u"):
            _bmm_fp32(
                score_tc_3d,
                u_3d,
                out=local_y_3d,
                precision=plan.precision,
                backend_identity=plan.contraction_backend_identity,
                loaded_backend_token=prepared.contraction_backend_token,
            )
        y_3d.add_(local_y_3d)
    else:
        with _record_hd_stage("hd.forward.local_score_u"):
            _record_hd_contraction(score_3d, u_3d, y_3d)
            with _record_hd_bmm_stage(score_3d, u_3d, y_3d):
                torch.baddbmm(
                    y_3d,
                    score_3d,
                    u_3d,
                    beta=1.0,
                    out=y_3d,
                )
    with _record_hd_stage("hd.forward.normalize"):
        _write_and_normalize_query_wave(
            y_wave,
            output,
            numerator,
            denominator,
            block_start=block_start,
            valid_tokens=valid_tokens,
            eps=eps,
            plan=plan,
        )


def _hd_parallel_block_gemm_impl(
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
    require_forward_only: bool,
    return_saved_carry: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    LoadedHdContractionBackendToken,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Run the shared plan-bound forward and optionally retain KV carry."""
    if require_forward_only:
        _require_forward_only_no_grad(q, k, v, a, b, c)
    eps = _require_eps(eps)
    q, k, v, layout, plan = _require_runtime_plan(
        q,
        k,
        v,
        layout=layout,
        plan=plan,
        require_forward_only=require_forward_only,
    )
    prepared = _prepare_feature_context(
        a,
        b,
        c,
        scale=scale,
        layout=layout,
        plan=plan,
        device=q.device,
        mode="forward",
    )
    contraction_backend_token = prepared.contraction_backend_token
    if contraction_backend_token is None:
        raise RuntimeError("prepared contraction backend token was released")
    with torch.no_grad(), prepared:
        block_totals = _allocate_planned(prepared, name="block_totals_carry")
        u_wave = _allocate_planned(prepared, name="forward_u_wave")
        block_totals_wave = _allocate_planned(
            prepared, name="forward_block_totals_wave"
        )
        phi_k_cache_name = {
            "forward": "forward_phi_k_cache",
            "backward": "saved_phi_k_tc",
        }.get(plan.key_retention)
        phi_k_cache = (
            _allocate_planned(prepared, name=phi_k_cache_name)
            if phi_k_cache_name is not None
            else None
        )

        for wave_index, block_start in enumerate(
            range(0, plan.number_blocks, plan.feature_wave_blocks)
        ):
            wave_blocks = min(
                plan.feature_wave_blocks, plan.number_blocks - block_start
            )
            with _record_hd_stage("hd.forward.k_feature_totals"):
                phi_k, u, valid_tokens = _prepare_key_value_wave(
                    k,
                    v,
                    prepared=prepared,
                    u_wave=u_wave,
                    block_start=block_start,
                    wave_blocks=wave_blocks,
                    plan=plan,
                    phi_k_cache_wave=(
                        phi_k_cache[wave_index] if phi_k_cache is not None else None
                    ),
                )
            with _record_hd_stage("hd.forward.block_total_gemm"):
                _write_block_totals_wave(
                    phi_k,
                    prepared=prepared,
                    u_wave=u,
                    block_totals=block_totals,
                    block_totals_wave=block_totals_wave,
                    block_start=block_start,
                    wave_blocks=wave_blocks,
                    valid_tokens=valid_tokens,
                    plan=plan,
                )

        del block_totals_wave
        scan_inclusive = _allocate_planned(prepared, name="forward_scan_inclusive")
        with _record_hd_stage("hd.forward.scan"):
            _exclusive_cumsum_block_totals(block_totals, scan_inclusive)
        del scan_inclusive

        local_score = _allocate_planned(prepared, name="forward_local_score")
        saved_local_score_tc = (
            _allocate_planned(prepared, name="saved_local_score_tc")
            if plan.save_local_score
            else None
        )
        local_score_tc = (
            _allocate_planned(prepared, name="forward_local_score_tc")
            if plan.precision == "bf16_tensorcore" and not plan.save_local_score
            else None
        )
        local_y = (
            _allocate_planned(prepared, name="forward_local_y")
            if plan.precision == "bf16_tensorcore"
            else None
        )
        phi_k_query_wave = (
            _allocate_planned(prepared, name="forward_phi_k_query_wave")
            if plan.gqa_ratio > 1
            else None
        )
        carry_query_wave = _allocate_planned(prepared, name="forward_carry_query_wave")
        u_query_wave = (
            _allocate_planned(prepared, name="forward_u_query_wave")
            if plan.gqa_ratio > 1
            else None
        )
        numerator = _allocate_planned(prepared, name="numerator")
        denominator = _allocate_planned(prepared, name="denominator")
        output = _allocate_planned(prepared, name="output")
        y_wave = _allocate_planned(prepared, name="forward_y_wave")

        for wave_index, block_start in enumerate(
            range(0, plan.number_blocks, plan.feature_wave_blocks)
        ):
            wave_blocks = min(
                plan.feature_wave_blocks, plan.number_blocks - block_start
            )
            _contract_query_wave(
                q,
                k,
                v,
                prepared=prepared,
                carry=block_totals,
                u_wave=u_wave,
                phi_k_query_wave=phi_k_query_wave,
                carry_query_wave=carry_query_wave,
                u_query_wave=u_query_wave,
                local_score=local_score,
                local_score_tc=(
                    saved_local_score_tc[wave_index]
                    if saved_local_score_tc is not None
                    else local_score_tc
                ),
                local_y=local_y,
                y_wave=y_wave,
                output=output,
                numerator=numerator,
                denominator=denominator,
                block_start=block_start,
                wave_blocks=wave_blocks,
                eps=eps,
                plan=plan,
                phi_k_cache_wave=(
                    phi_k_cache[wave_index] if phi_k_cache is not None else None
                ),
            )

        with _record_hd_stage("hd.forward.denominator_check"):
            _require_positive_denominator(denominator)
        query_side = any(
            (
                plan.requested_gradient_mask[0],
                *plan.requested_gradient_mask[3:],
            )
        )
        saved_carry = block_totals if return_saved_carry and query_side else None
        saved_phi_k_tc = phi_k_cache if plan.key_retention == "backward" else None
        return (
            output,
            numerator,
            denominator,
            saved_carry,
            contraction_backend_token,
            saved_local_score_tc,
            saved_phi_k_tc,
        )


def hd_parallel_block_gemm(
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
    """Run one explicit forward-only causal Block-GEMM invocation."""
    output, numerator, denominator, _, _, _, _ = _hd_parallel_block_gemm_impl(
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
        require_forward_only=True,
        return_saved_carry=False,
    )
    return output, numerator, denominator


def _saved_hd_parallel_block_gemm(
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
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    LoadedHdContractionBackendToken,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Run one gradient-plan forward and return only requested saved state."""
    return _hd_parallel_block_gemm_impl(
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
        require_forward_only=False,
        return_saved_carry=True,
    )


__all__: tuple[str, ...] = ()
