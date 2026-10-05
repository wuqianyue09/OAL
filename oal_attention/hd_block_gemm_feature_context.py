"""Prepared invocation state and plan-bound storage for packed HD features."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from numbers import Real
from typing import Literal
import weakref

import torch

from .hd_block_gemm_plan import HDParallelBlockPlan
from .hd_block_gemm_contracts import _require_implemented_query_operator_identity
from .hd_block_gemm_cache import (
    CanonicalPairLayout,
    _PairCacheAdmissionLease,
    _abandon_pair_cache_admission_lease,
    _acquire_pair_cache_admission_lease,
    _materialize_pair_layout,
    _release_pair_cache_admission_lease,
    _validate_pair_cache_admission_lease,
    canonical_pair_layout,
)
from .hd_cublas_compat import (
    LoadedHdContractionBackendToken,
    prepare_backend_token,
)
from .hd_block_gemm_profiling import (
    _planned_allocation_category,
    _record_hd_allocation,
)

_SUPPORTED_FLOAT_DTYPES = frozenset(
    (torch.float16, torch.bfloat16, torch.float32, torch.float64)
)
_FORWARD = "forward"
_BACKWARD = "backward"
_NEW = "NEW"
_ACTIVE = "ACTIVE"
_FAILED = "FAILED"
_CLOSED = "CLOSED"
_COEFFICIENT_IDLE = "IDLE"
_COEFFICIENT_ACCUMULATING = "ACCUMULATING"
_COEFFICIENT_FINALIZED = "FINALIZED"


@dataclass(frozen=True)
class _RawFeatureValues:
    a: torch.Tensor | None
    b: torch.Tensor | None
    c: torch.Tensor | None
    scale: float | None


@dataclass(frozen=True)
class _PreparedFeatureValues:
    a: torch.Tensor | None
    b: torch.Tensor | None
    c: torch.Tensor | None
    scale: torch.Tensor | None


@dataclass(frozen=True)
class _PairMetadata:
    rows: torch.Tensor
    columns: torch.Tensor
    multiplicity: torch.Tensor


@dataclass(frozen=True)
class _ForwardFeatureWorkspace:
    q_input_work: torch.Tensor | None
    k_input_work: torch.Tensor | None
    pair_work: torch.Tensor | None
    pair_second_work: torch.Tensor | None
    phi_k_output: torch.Tensor | None
    phi_q_output: torch.Tensor | None


@dataclass(frozen=True)
class _CoefficientWorkspace:
    a_block_partials: torch.Tensor | None
    b_block_partials: torch.Tensor | None
    c_block_partials: torch.Tensor | None
    a_output: torch.Tensor | None
    b_output: torch.Tensor | None
    c_output: torch.Tensor | None


@dataclass(frozen=True)
class _GenericQueryFoldWorkspace:
    """Scratch owned solely by the generic query-gradient fold."""

    pair_source: torch.Tensor | None
    pair_second: torch.Tensor | None
    linear_source: torch.Tensor | None


@dataclass(frozen=True)
class _BackwardFeatureWorkspace:
    feature_pair_work: torch.Tensor | None
    feature_pair_second_work: torch.Tensor | None
    feature_k_input_work: torch.Tensor | None
    phi_k_output: torch.Tensor | None
    phi_q_output: torch.Tensor | None
    query_input_work: torch.Tensor | None
    generic_query_fold: _GenericQueryFoldWorkspace | None
    query_output: torch.Tensor | None
    key_input_work: torch.Tensor | None
    key_pair_source: torch.Tensor | None
    key_output: torch.Tensor | None
    coefficients: _CoefficientWorkspace


def _finalize_abandoned_lease(lease: _PairCacheAdmissionLease) -> None:
    """Best-effort death-owner cleanup; it owns no context reference."""
    _abandon_pair_cache_admission_lease(lease)


@dataclass
class _PreparedFeatureContext:
    """One explicit cache lease with transactionally committed small bundles."""

    plan: HDParallelBlockPlan
    layout: CanonicalPairLayout
    device: torch.device
    mode: Literal["forward", "backward"]
    raw_values: _RawFeatureValues | None
    contraction_backend_token: LoadedHdContractionBackendToken | None
    state: str = _NEW
    lease: _PairCacheAdmissionLease | None = None
    lease_finalizer: weakref.finalize | None = None
    release_error: BaseException | None = None
    values: _PreparedFeatureValues | None = None
    pairs: _PairMetadata | None = None
    forward_workspace: _ForwardFeatureWorkspace | None = None
    backward_workspace: _BackwardFeatureWorkspace | None = None
    coefficient_state: str = _COEFFICIENT_IDLE
    coefficient_coverage: list[bool] | None = None

    def __enter__(self) -> _PreparedFeatureContext:
        if self.state == _CLOSED:
            raise RuntimeError("prepared feature context is closed")
        if self.state == _FAILED:
            raise RuntimeError("prepared feature context has failed")
        if self.state != _NEW:
            raise RuntimeError("prepared feature context is already entered")
        raw_values = _require_raw_values(self)
        _require_no_grad_tensors(raw_values.a, raw_values.b, raw_values.c)
        lease: _PairCacheAdmissionLease | None = None
        try:
            lease = _acquire_pair_cache_admission_lease(self.plan)
            finalizer = weakref.finalize(self, _finalize_abandoned_lease, lease)
        except BaseException:
            self.state = _FAILED
            if lease is not None:
                _abandon_pair_cache_admission_lease(lease)
            self._clear_storage()
            raise
        self.lease = lease
        self.lease_finalizer = finalizer
        self.state = _ACTIVE
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> bool:
        del exception_type, traceback
        if exception is None:
            self.close()
        else:
            try:
                self.close()
            except BaseException as release_error:
                self.release_error = release_error
        return False

    def _detach_finalizer(self) -> None:
        finalizer = self.lease_finalizer
        if finalizer is not None and finalizer.alive:
            finalizer.detach()
        self.lease_finalizer = None

    def _release_live_lease(self) -> None:
        lease = self.lease
        if lease is None:
            return
        _release_pair_cache_admission_lease(lease)
        self.lease = None
        self.release_error = None
        self._detach_finalizer()

    def _clear_storage(self) -> None:
        self.raw_values = None
        self.values = None
        self.pairs = None
        self.forward_workspace = None
        self.backward_workspace = None
        self.coefficient_coverage = None
        self.contraction_backend_token = None

    def close(self) -> None:
        if self.state == _CLOSED:
            return
        if self.state == _NEW:
            self._clear_storage()
            self.state = _CLOSED
            return
        if self.state not in (_ACTIVE, _FAILED):
            raise RuntimeError("prepared feature context has invalid state")
        was_active = self.state == _ACTIVE
        try:
            self._release_live_lease()
        except BaseException as error:
            self.release_error = error
            self.state = _FAILED
            self._clear_storage()
            raise
        self._clear_storage()
        if was_active:
            self.state = _CLOSED

    def _fail(self) -> None:
        self.state = _FAILED
        try:
            self._release_live_lease()
        except BaseException as error:
            self.release_error = error
        finally:
            self._clear_storage()


def _require_plan(plan: object) -> HDParallelBlockPlan:
    if not isinstance(plan, HDParallelBlockPlan):
        raise TypeError("plan must be an HDParallelBlockPlan")
    return plan


def _require_layout(
    layout: object,
    *,
    plan: HDParallelBlockPlan,
) -> CanonicalPairLayout:
    if not isinstance(layout, CanonicalPairLayout):
        raise TypeError("layout must be a CanonicalPairLayout")
    if layout != canonical_pair_layout(plan.head_dimension):
        raise ValueError("layout must be canonical for the plan head dimension")
    if layout.layout_id != plan.pair_layout_id:
        raise ValueError("layout does not match the plan")
    return layout


def _require_device(device: object, *, plan: HDParallelBlockPlan) -> torch.device:
    if not isinstance(device, (str, torch.device)):
        raise TypeError("device must be a string or torch.device")
    try:
        canonical = torch.device(device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("device is not a valid torch device") from error
    if str(canonical) != plan.device:
        raise ValueError("device does not match the plan")
    return canonical


def _require_scale(scale: object) -> float:
    if not isinstance(scale, Real) or isinstance(scale, bool):
        raise TypeError("scale must be a real number")
    try:
        numeric = float(scale)
    except (OverflowError, ValueError) as error:
        raise ValueError("scale must be finite and representable in FP32") from error
    if not math.isfinite(numeric) or abs(numeric) > torch.finfo(torch.float32).max:
        raise ValueError("scale must be finite and representable in FP32")
    if numeric != 0.0 and abs(numeric) <= math.ldexp(1.0, -150):
        raise ValueError("scale must be finite and representable in FP32")
    return numeric


def _require_coefficient(
    coefficient: object,
    *,
    name: str,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(coefficient, torch.Tensor):
        raise ValueError(f"{name} coefficient is required")
    if coefficient.layout != torch.strided:
        raise ValueError(f"{name} must use strided layout")
    if tuple(coefficient.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if coefficient.dtype not in _SUPPORTED_FLOAT_DTYPES:
        raise TypeError(f"{name} must use a supported floating-point dtype")
    if coefficient.device != device:
        raise ValueError(f"{name} device must match the plan")
    return coefficient


def _require_no_grad_tensors(*tensors: torch.Tensor | None) -> None:
    if torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad for tensor in tensors
    ):
        raise RuntimeError(
            "packed feature helpers require a no-grad custom autograd region"
        )


def _require_raw_values(prepared: _PreparedFeatureContext) -> _RawFeatureValues:
    raw_values = prepared.raw_values
    if raw_values is None:
        raise RuntimeError("prepared feature context storage has been released")
    return raw_values


def _prepare_feature_context(
    a: object,
    b: object,
    c: object,
    *,
    scale: object,
    layout: CanonicalPairLayout,
    plan: HDParallelBlockPlan,
    device: torch.device,
    mode: Literal["forward", "backward"],
) -> _PreparedFeatureContext:
    """Validate an invocation before acquiring a lease or allocating storage."""
    plan = _require_plan(plan)
    _require_implemented_query_operator_identity(
        query_feature_impl=plan.query_feature_impl,
        query_fold_impl=plan.query_fold_impl,
        query_gradient_flow=plan.query_gradient_flow,
        query_producer_fold_strategy=plan.query_producer_fold_strategy,
        query_consumer_stages=plan.query_consumer_stages,
    )
    layout = _require_layout(layout, plan=plan)
    device = _require_device(device, plan=plan)
    if mode not in (_FORWARD, _BACKWARD):
        raise ValueError("mode must be 'forward' or 'backward'")
    need_q, need_k, need_v, _, need_b, need_c = plan.requested_gradient_mask
    right_scan = need_k or need_v
    need_a_value = mode == _FORWARD or (right_scan and a is not None)
    need_b_value = mode == _FORWARD or need_q or (right_scan and b is not None)
    need_c_value = mode == _FORWARD or need_q or (right_scan and c is not None)
    need_scale_value = (
        mode == _FORWARD
        or need_q
        or need_b
        or need_c
        or (right_scan and scale is not None)
    )
    raw_values = _RawFeatureValues(
        a=(
            _require_coefficient(a, name="A", shape=(plan.query_heads,), device=device)
            if need_a_value
            else None
        ),
        b=(
            _require_coefficient(
                b,
                name="B",
                shape=(plan.query_heads, plan.head_dimension),
                device=device,
            )
            if need_b_value
            else None
        ),
        c=(
            _require_coefficient(
                c,
                name="C",
                shape=(plan.query_heads, plan.pair_count),
                device=device,
            )
            if need_c_value
            else None
        ),
        scale=_require_scale(scale) if need_scale_value else None,
    )
    _require_no_grad_tensors(raw_values.a, raw_values.b, raw_values.c)
    contraction_backend_token = prepare_backend_token(
        plan.contraction_backend_identity,
        device,
    )
    return _PreparedFeatureContext(
        plan=plan,
        layout=layout,
        device=device,
        mode=mode,
        raw_values=raw_values,
        contraction_backend_token=contraction_backend_token,
    )


def _require_prepared(
    prepared: object,
    *,
    mode: Literal["forward", "backward"],
) -> _PreparedFeatureContext:
    if not isinstance(prepared, _PreparedFeatureContext):
        raise TypeError("prepared must be a prepared feature context")
    if prepared.mode != mode:
        raise ValueError(f"prepared context must use {mode} mode")
    if prepared.state == _NEW:
        raise RuntimeError("prepared feature context must be explicitly entered")
    if prepared.state == _FAILED:
        raise RuntimeError("prepared feature context has failed")
    if prepared.state == _CLOSED:
        raise RuntimeError("prepared feature context is closed")
    token = prepared.contraction_backend_token
    if token is None:
        raise RuntimeError("prepared contraction backend token was released")
    if token.identity != prepared.plan.contraction_backend_identity:
        raise RuntimeError(
            "prepared contraction backend identity does not match the plan"
        )
    return prepared


def _planned_empty(prepared: _PreparedFeatureContext, name: str) -> torch.Tensor:
    """Allocate the plan's exact storage under the active invocation lease."""
    prepared = _require_prepared(prepared, mode=prepared.mode)
    lease = prepared.lease
    if lease is None:
        raise RuntimeError("prepared feature context has no active cache lease")
    _validate_pair_cache_admission_lease(lease, prepared.plan)
    buffer = prepared.plan.buffer(name)
    # HDLogicalBuffer validates dtype names, including normalization's bool mask.
    dtype = getattr(torch, buffer.dtype)
    return _record_hd_allocation(
        name,
        torch.empty(buffer.shape, dtype=dtype, device=prepared.device),
        category=_planned_allocation_category(name),
    )


def _copy_into(destination: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    return destination.copy_(source)


def _ensure_prepared_values(prepared: _PreparedFeatureContext) -> None:
    raw = _require_raw_values(prepared)
    _require_no_grad_tensors(raw.a, raw.b, raw.c)
    if prepared.values is not None:
        return
    try:
        local_a = _planned_empty(prepared, "logical_a") if raw.a is not None else None
        if local_a is not None:
            _copy_into(local_a, raw.a)
        local_b = _planned_empty(prepared, "logical_b") if raw.b is not None else None
        if local_b is not None:
            _copy_into(local_b, raw.b)
        local_c = _planned_empty(prepared, "logical_c") if raw.c is not None else None
        if local_c is not None:
            _copy_into(local_c, raw.c)
        local_scale = (
            _planned_empty(prepared, "logical_scale") if raw.scale is not None else None
        )
        if local_scale is not None:
            local_scale.fill_(raw.scale)
        local_values = _PreparedFeatureValues(
            a=local_a,
            b=local_b,
            c=local_c,
            scale=local_scale,
        )
    except Exception:
        prepared._fail()
        raise
    prepared.values = local_values


def _ensure_pairs(prepared: _PreparedFeatureContext) -> None:
    if prepared.pairs is not None:
        return
    try:
        rows, columns, multiplicity = _materialize_pair_layout(
            prepared.layout, prepared.device, plan=prepared.plan
        )
        local_pairs = _PairMetadata(rows, columns, multiplicity)
    except Exception:
        prepared._fail()
        raise
    prepared.pairs = local_pairs


def _ensure_forward_workspace(prepared: _PreparedFeatureContext) -> None:
    if prepared.forward_workspace is not None:
        return
    try:
        buffer_names = {buffer.name for buffer in prepared.plan.logical_buffers}
        local_workspace = _ForwardFeatureWorkspace(
            q_input_work=(
                _planned_empty(prepared, "forward_phi_q_input_work")
                if prepared.plan.query_feature_impl == "generic_materialized"
                else None
            ),
            k_input_work=(
                _planned_empty(prepared, "forward_phi_k_input_work")
                if "forward_phi_k_input_work" in buffer_names
                else None
            ),
            pair_work=(
                _planned_empty(prepared, "forward_pair_work")
                if "forward_pair_work" in buffer_names
                else None
            ),
            pair_second_work=(
                _planned_empty(prepared, "forward_pair_second_work")
                if "forward_pair_second_work" in buffer_names
                else None
            ),
            phi_k_output=(
                _planned_empty(prepared, "forward_phi_k_wave")
                if "forward_phi_k_wave" in buffer_names
                else None
            ),
            phi_q_output=(
                _planned_empty(prepared, "forward_phi_q_wave")
                if "forward_phi_q_wave" in buffer_names
                else None
            ),
        )
    except Exception:
        prepared._fail()
        raise
    prepared.forward_workspace = local_workspace


def _empty_coefficient_workspace() -> _CoefficientWorkspace:
    return _CoefficientWorkspace(
        a_block_partials=None,
        b_block_partials=None,
        c_block_partials=None,
        a_output=None,
        b_output=None,
        c_output=None,
    )


def _empty_backward_workspace() -> _BackwardFeatureWorkspace:
    return _BackwardFeatureWorkspace(
        feature_pair_work=None,
        feature_pair_second_work=None,
        feature_k_input_work=None,
        phi_k_output=None,
        phi_q_output=None,
        query_input_work=None,
        generic_query_fold=None,
        query_output=None,
        key_input_work=None,
        key_pair_source=None,
        key_output=None,
        coefficients=_empty_coefficient_workspace(),
    )


def _allocate_missing(
    prepared: _PreparedFeatureContext,
    existing: torch.Tensor | None,
    requested: bool,
    name: str,
) -> torch.Tensor | None:
    if existing is not None or not requested:
        return existing
    return _planned_empty(prepared, name)


def _ensure_backward_workspace(
    prepared: _PreparedFeatureContext,
    *,
    feature_key: bool = False,
    feature_query: bool = False,
    query_fold: bool = False,
    key_fold: bool = False,
    coefficients: bool = False,
) -> None:
    current = prepared.backward_workspace or _empty_backward_workspace()
    need_q, need_k, _, need_a, need_b, need_c = prepared.plan.requested_gradient_mask
    needs_materialized_query_input = (
        prepared.plan.query_gradient_flow == "materialized"
        and getattr(prepared.plan, "query_fold_input", "staged_fp32") == "staged_fp32"
        and (need_q or need_b or need_c)
    )
    needs_generic_query_feature_input = (
        prepared.plan.query_feature_impl == "generic_materialized"
        and (need_k or prepared.plan.requested_gradient_mask[2])
    )
    needs_feature_pair_scratch = (
        feature_key and prepared.plan.key_feature_impl == "generic_materialized"
    ) or (feature_query and prepared.plan.query_feature_impl == "generic_materialized")
    try:
        coefficient_workspace = current.coefficients
        if coefficients:
            coefficient_workspace = replace(
                coefficient_workspace,
                a_block_partials=_allocate_missing(
                    prepared,
                    coefficient_workspace.a_block_partials,
                    need_a,
                    "backward_coefficient_a_block_partials",
                ),
                b_block_partials=_allocate_missing(
                    prepared,
                    coefficient_workspace.b_block_partials,
                    need_b,
                    "backward_coefficient_b_block_partials",
                ),
                c_block_partials=_allocate_missing(
                    prepared,
                    coefficient_workspace.c_block_partials,
                    need_c,
                    "backward_coefficient_c_block_partials",
                ),
                a_output=_allocate_missing(
                    prepared,
                    coefficient_workspace.a_output,
                    need_a,
                    "grad_a",
                ),
                b_output=_allocate_missing(
                    prepared,
                    coefficient_workspace.b_output,
                    need_b,
                    "grad_b",
                ),
                c_output=_allocate_missing(
                    prepared,
                    coefficient_workspace.c_output,
                    need_c,
                    "grad_c",
                ),
            )
        generic_query_fold = current.generic_query_fold
        if query_fold and prepared.plan.query_fold_impl == "generic_materialized":
            generic_current = generic_query_fold or _GenericQueryFoldWorkspace(
                pair_source=None,
                pair_second=None,
                linear_source=None,
            )
            generic_query_fold = replace(
                generic_current,
                pair_source=_allocate_missing(
                    prepared,
                    generic_current.pair_source,
                    need_q or need_c,
                    "backward_query_fold_source",
                ),
                pair_second=_allocate_missing(
                    prepared,
                    generic_current.pair_second,
                    need_c,
                    "backward_query_fold_second",
                ),
                linear_source=_allocate_missing(
                    prepared,
                    generic_current.linear_source,
                    need_b,
                    "backward_query_fold_linear_source",
                ),
            )
        local_workspace = replace(
            current,
            feature_pair_work=_allocate_missing(
                prepared,
                current.feature_pair_work,
                needs_feature_pair_scratch,
                "backward_feature_pair_work",
            ),
            feature_pair_second_work=_allocate_missing(
                prepared,
                current.feature_pair_second_work,
                needs_feature_pair_scratch
                and prepared.plan.precision == "bf16_tensorcore",
                "backward_feature_pair_second_work",
            ),
            feature_k_input_work=_allocate_missing(
                prepared,
                current.feature_k_input_work,
                feature_key
                and prepared.plan.precision == "bf16_tensorcore"
                and prepared.plan.key_feature_impl == "generic_materialized",
                "backward_feature_k_input_work",
            ),
            phi_k_output=_allocate_missing(
                prepared,
                current.phi_k_output,
                feature_key,
                "backward_phi_k_wave",
            ),
            phi_q_output=_allocate_missing(
                prepared,
                current.phi_q_output,
                feature_query,
                "backward_phi_q_wave",
            ),
            query_input_work=_allocate_missing(
                prepared,
                current.query_input_work,
                needs_generic_query_feature_input
                or (query_fold and needs_materialized_query_input),
                "backward_query_input_work",
            ),
            generic_query_fold=generic_query_fold,
            query_output=_allocate_missing(
                prepared,
                current.query_output,
                query_fold and need_q,
                "backward_query_fold_output",
            ),
            key_input_work=_allocate_missing(
                prepared,
                current.key_input_work,
                key_fold
                and need_k
                and prepared.plan.key_fold_impl == "generic_materialized",
                "backward_key_input_work",
            ),
            key_pair_source=_allocate_missing(
                prepared,
                current.key_pair_source,
                key_fold
                and need_k
                and prepared.plan.key_fold_impl == "generic_materialized",
                "backward_key_fold_source",
            ),
            key_output=_allocate_missing(
                prepared,
                current.key_output,
                key_fold and need_k,
                "backward_key_fold_output",
            ),
            coefficients=coefficient_workspace,
        )
    except Exception:
        prepared._fail()
        raise
    prepared.backward_workspace = local_workspace


def _release_backward_feature_outputs(
    prepared: _PreparedFeatureContext,
    *,
    key: bool = False,
    query: bool = False,
) -> None:
    """Drop feature-output references when their planned stage has ended."""
    prepared = _require_prepared(prepared, mode="backward")
    current = prepared.backward_workspace
    if current is None or not (key or query):
        return
    prepared.backward_workspace = replace(
        current,
        phi_k_output=None if key else current.phi_k_output,
        phi_q_output=None if query else current.phi_q_output,
    )


def _wave_capacity(plan: HDParallelBlockPlan) -> int:
    return plan.feature_wave_blocks * plan.token_block


def _require_feature_wave(
    tensor: object,
    *,
    name: str,
    head_count: int,
    plan: HDParallelBlockPlan,
) -> tuple[torch.Tensor, int]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tensor.layout != torch.strided:
        raise ValueError(f"{name} must use strided layout")
    if tensor.ndim < 3:
        raise ValueError(f"{name} must have shape [B,H,...,D]")
    if any(dimension <= 0 for dimension in tensor.shape):
        raise ValueError(f"{name} dimensions must be positive")
    if tensor.shape[0] != plan.batch_size:
        raise ValueError(f"{name} batch dimension does not match the plan")
    if tensor.shape[1] != head_count:
        raise ValueError(f"{name} head dimension does not match the plan")
    if tensor.shape[-1] != plan.head_dimension:
        raise ValueError(f"{name} feature dimension does not match the plan")
    if tensor.dtype not in _SUPPORTED_FLOAT_DTYPES:
        raise TypeError(f"{name} must use a supported floating-point dtype")
    dtype_name = str(tensor.dtype).removeprefix("torch.")
    if dtype_name != plan.input_dtype:
        raise ValueError(f"{name} dtype does not match the plan")
    if str(tensor.device) != plan.device:
        raise ValueError(f"{name} device does not match the plan")
    token_count = math.prod(tensor.shape[2:-1])
    if token_count > _wave_capacity(plan):
        raise ValueError(f"{name} exceeds the planned feature wave capacity")
    return tensor, token_count


def _require_feature_gradient(
    gradient: object,
    *,
    name: str,
    head_count: int,
    plan: HDParallelBlockPlan,
) -> tuple[torch.Tensor, int]:
    if not isinstance(gradient, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if gradient.layout != torch.strided:
        raise ValueError(f"{name} must use strided layout")
    if gradient.ndim < 3:
        raise ValueError(f"{name} must have shape [B,H,...,M]")
    if any(dimension <= 0 for dimension in gradient.shape):
        raise ValueError(f"{name} dimensions must be positive")
    if gradient.shape[0] != plan.batch_size:
        raise ValueError(f"{name} batch dimension does not match the plan")
    if gradient.shape[1] != head_count:
        raise ValueError(f"{name} head dimension does not match the plan")
    if gradient.shape[-1] != plan.physical_feature_dimension:
        raise ValueError(f"{name} feature dimension does not match the plan")
    if gradient.dtype != torch.float32:
        raise ValueError(f"{name} must use planned FP32 storage")
    if str(gradient.device) != plan.device:
        raise ValueError(f"{name} device does not match the plan")
    token_count = math.prod(gradient.shape[2:-1])
    if token_count > _wave_capacity(plan):
        raise ValueError(f"{name} exceeds the planned feature wave capacity")
    return gradient, token_count


def _require_coefficient_block_wave(
    d_phi_q: torch.Tensor,
    q: torch.Tensor | None,
    *,
    block_start: object,
    needs_input_q: bool,
    plan: HDParallelBlockPlan,
) -> tuple[int, int]:
    if d_phi_q.ndim != 5 or d_phi_q.shape[3] != plan.token_block:
        raise ValueError("coefficient dPhiQ wave must have shape [B,Hq,WB,BT,M]")
    wave_blocks = d_phi_q.shape[2]
    if wave_blocks > plan.feature_wave_blocks:
        raise ValueError("coefficient dPhiQ wave exceeds planned wave blocks")
    if needs_input_q and (q is None or q.shape[:-1] != d_phi_q.shape[:-1]):
        raise ValueError("coefficient Q wave must have shape [B,Hq,WB,BT,D]")
    if not isinstance(block_start, int) or isinstance(block_start, bool):
        raise TypeError("block_start must be a nonnegative integer")
    if block_start < 0 or block_start + wave_blocks > plan.number_blocks:
        raise ValueError("coefficient gradient block range is out of bounds")
    return block_start, wave_blocks


def _wave_view(
    storage: torch.Tensor,
    *,
    wave_shape: torch.Size,
    token_count: int,
    feature_dimension: int,
) -> torch.Tensor:
    batch_size, head_count, *token_shape = wave_shape
    if storage.shape[-1] != feature_dimension:
        raise ValueError("planned storage row pitch does not match requested view")
    return storage[:, :head_count, :token_count, :].view(
        batch_size, head_count, *token_shape, feature_dimension
    )


def _copy_wave_to_work(
    storage: torch.Tensor,
    source: torch.Tensor,
    *,
    token_count: int,
) -> torch.Tensor:
    view = _wave_view(
        storage,
        wave_shape=source.shape[:-1],
        token_count=token_count,
        feature_dimension=source.shape[-1],
    )
    _copy_into(view, source)
    return view


def _head_scalar_view(values: torch.Tensor, *, wave_ndim: int) -> torch.Tensor:
    return values.view(1, values.shape[0], *([1] * (wave_ndim - 3)))


def _head_feature_view(values: torch.Tensor, *, wave_ndim: int) -> torch.Tensor:
    return values.view(1, values.shape[0], *([1] * (wave_ndim - 3)), values.shape[1])


def _pair_metadata_view(values: torch.Tensor, *, wave_ndim: int) -> torch.Tensor:
    return values.view(*([1] * (wave_ndim - 1)), values.numel())


def _scatter_pair_source(
    output: torch.Tensor,
    index: torch.Tensor,
    source: torch.Tensor,
) -> None:
    scatter_index = _pair_metadata_view(index, wave_ndim=source.ndim).expand_as(source)
    output.scatter_add_(-1, scatter_index, source)


def _require_scatter_mode(device: torch.device) -> None:
    if device.type == "cuda" and torch.are_deterministic_algorithms_enabled():
        raise RuntimeError(
            "controlled scatter reduction does not support CUDA deterministic mode"
        )


def _new_coefficient_coverage(number_blocks: int) -> list[bool]:
    return [False] * number_blocks


__all__: tuple[str, ...] = ()
