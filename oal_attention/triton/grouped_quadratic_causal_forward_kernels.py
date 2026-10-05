"""Planned macro-token CUDA kernels for causal Hadamard-diag attention.

Each KV head owns its FP32 H0/H1/H2 carry.  For every macro range the H2
kernel has one grid dimension over every canonical packed-pair block; those
programs write exclusive bounded partials.  A single reducer then consumes
pair blocks in ascending canonical order, with no floating-point atomic and
without a host-side pair-wave loop.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Final

import torch

from .grouped_quadratic_causal_common import (
    KernelGeometry,
    KernelPlan,
    PHYSICAL_PATH_IDENTIFIER,
    build_kernel_plan,
    collect_cuda_device_properties,
    cuda_device_index,
)
from .grouped_quadratic_production_witness import (
    ProductionWitnessSession,
    active_production_witness_session,
)

try:
    import triton
    import triton.language as tl
except (ImportError, ModuleNotFoundError):
    triton = None
    tl = None


_FORWARD_WORKSPACE_ALLOCATION_NAMES: Final[tuple[str, ...]] = (
    "forward_state",
    "forward_pair_partial",
    "output",
    "numerator",
    "denominator",
    "denominator_partial",
)
_LAST_FORWARD_PLAN_ID: str | None = None
_FORWARD_PRODUCTION_WITNESS_CAPTURE_NAMES: Final[tuple[str, ...]] = (
    "forward.h0_increment",
    "forward.h1_increment",
    "forward.h2_increment",
    "forward.inclusive_state",
    "forward.diagonal_contraction",
)


def triton_is_available() -> bool:
    """Return whether the planned CUDA forward kernels can be launched."""
    return triton is not None and tl is not None


def forward_production_witness_capture_names() -> tuple[str, ...]:
    """Return the fixed scalar probes written only by an evidence session."""
    return _FORWARD_PRODUCTION_WITNESS_CAPTURE_NAMES


def planned_forward_macro_ranges(plan: KernelPlan) -> tuple[tuple[int, int], ...]:
    """Return fixed-order macro ranges; packed-pair tiling is irrelevant here."""
    if not isinstance(plan, KernelPlan):
        raise TypeError("planned forward macros require a KernelPlan")
    if plan.physical_path != PHYSICAL_PATH_IDENTIFIER or not plan.geometry.causal:
        raise ValueError("planned forward macros require the causal physical path")
    macro_token_block = plan.specialization_axes["macro_token_block"]
    if not isinstance(macro_token_block, int) or macro_token_block <= 0:
        raise ValueError("kernel plan has no positive macro_token_block")
    return tuple(
        (start, min(macro_token_block, plan.geometry.N - start))
        for start in range(0, plan.geometry.N, macro_token_block)
    )


def launch_planned_forward_macros(
    plan: KernelPlan,
    launch_macro: Callable[..., None],
) -> None:
    """Invoke one launch boundary per macro, in increasing token order."""
    if not callable(launch_macro):
        raise TypeError("launch_macro must be callable")
    for chunk_start, chunk_length in planned_forward_macro_ranges(plan):
        launch_macro(chunk_start=chunk_start, chunk_length=chunk_length)


def forward_workspace_allocation_names(plan: KernelPlan) -> tuple[str, ...]:
    """Return exactly the physical, forward-lifetime plan allocations."""
    if not isinstance(plan, KernelPlan):
        raise TypeError("forward workspace requires a KernelPlan")
    for name in _FORWARD_WORKSPACE_ALLOCATION_NAMES:
        allocation = plan.workspace.allocation(name)
        if allocation.alias_of is not None:
            raise ValueError(f"forward workspace allocation {name} cannot be an alias")
    return _FORWARD_WORKSPACE_ALLOCATION_NAMES


def _torch_dtype(dtype_name: str) -> torch.dtype:
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"unsupported planned tensor dtype: {dtype_name}")
    return dtype


def allocate_planned_forward_workspace(
    plan: KernelPlan,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Allocate only exact ``WorkspacePlan`` records used by this forward path."""
    buffers: dict[str, torch.Tensor] = {}
    for name in forward_workspace_allocation_names(plan):
        allocation = plan.workspace.allocation(name)
        kwargs = {"device": device, "dtype": _torch_dtype(allocation.dtype)}
        buffers[name] = (
            torch.zeros(allocation.shape, **kwargs)
            if name == "forward_state"
            else torch.empty(allocation.shape, **kwargs)
        )
    return buffers


@lru_cache(maxsize=1)
def _source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _stride_class(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
) -> str:
    if not all(
        tensor.is_contiguous() for tensor in (q, k, v, dim_groups, linear, quadratic)
    ):
        raise ValueError("planned grouped causal CUDA path requires contiguous tensors")
    if constant.ndim != 1 or constant.stride(0) <= 0:
        raise ValueError(
            "planned grouped causal constant must be a positive-stride [Hq] view"
        )
    return "contiguous" if constant.is_contiguous() else "constant_head_strided"


def _validate_triton_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
) -> None:
    if not triton_is_available():
        raise RuntimeError("planned grouped causal forward requires Triton")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise RuntimeError("planned grouped causal forward requires CUDA tensors")
    if (
        q.ndim != 4
        or k.ndim != 4
        or v.ndim != 4
        or q.shape[0] != k.shape[0]
        or q.shape[0] != v.shape[0]
        or q.shape[2] != k.shape[2]
        or q.shape[2] != v.shape[2]
    ):
        raise ValueError(
            "planned grouped causal forward requires compatible rank-4 Q/K/V"
        )
    if k.shape[1] != v.shape[1]:
        raise ValueError(
            "planned grouped causal forward requires matching K/V head counts"
        )
    if not all(dimension > 0 for tensor in (q, k, v) for dimension in tensor.shape):
        raise ValueError(
            "planned grouped causal forward requires positive Q/K/V dimensions"
        )
    if (
        q.dtype not in {torch.float16, torch.bfloat16}
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise TypeError(
            "planned grouped causal forward supports matching float16/bfloat16 Q/K/V"
        )
    if q.shape[-1] != k.shape[-1] or not 1 <= q.shape[-1] <= 64:
        raise ValueError("planned grouped causal forward supports D through 64")
    if not 1 <= v.shape[-1] <= 64:
        raise ValueError("planned grouped causal forward supports DV through 64")
    if q.shape[1] % k.shape[1] or q.shape[1] // k.shape[1] > 16:
        raise ValueError(
            "planned grouped causal forward supports native GQA ratios through 16"
        )
    if dim_groups.dtype != torch.int32:
        raise TypeError("dim_groups must be canonical int32")
    if any(tensor.dtype != torch.float32 for tensor in (constant, linear, quadratic)):
        raise TypeError(
            "expanded constant/linear/quadratic coefficients must be float32"
        )
    gmax = linear.shape[-1] if linear.ndim == 2 else 0
    if (
        constant.shape != (q.shape[1],)
        or linear.shape != (q.shape[1], gmax)
        or quadratic.shape != (q.shape[1], gmax, gmax)
    ):
        raise ValueError(
            "planned grouped causal forward requires expanded [Hq]/[Hq,G]/[Hq,G,G] coefficients"
        )
    if dim_groups.ndim == 1:
        valid_groups_shape = dim_groups.shape == (q.shape[-1],)
    elif dim_groups.ndim == 2:
        valid_groups_shape = dim_groups.shape == (q.shape[1], q.shape[-1])
    else:
        valid_groups_shape = False
    if not valid_groups_shape:
        raise ValueError(
            "dim_groups must have canonical shared [D] or per-head [Hq,D] shape"
        )
    if not all(
        tensor.device == q.device
        for tensor in (k, v, dim_groups, constant, linear, quadratic)
    ):
        raise ValueError("planned grouped causal forward tensors must share one device")
    if (
        gmax <= 0
        or bool(torch.any(dim_groups < 0))
        or bool(torch.any(dim_groups >= gmax))
    ):
        raise ValueError("dim_groups must index the expanded coefficient groups")
    _stride_class(q, k, v, dim_groups, constant, linear, quadratic)


def _runtime_forward_geometry(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    *,
    workspace_budget_bytes: int | None,
) -> KernelGeometry:
    """Describe the live CUDA/toolchain/tensor tuple without allocating workspace."""
    assert triton is not None
    properties = torch.cuda.get_device_properties(q.device)
    hardware = collect_cuda_device_properties(
        properties,
        device_index=cuda_device_index(q.device),
        triton_module=triton,
    )
    return KernelGeometry(
        compute_capability=(properties.major, properties.minor),
        **hardware,
        torch_version=torch.__version__,
        triton_version=str(getattr(triton, "__version__", "unknown")),
        source_hash=_source_hash(),
        dtype=q.dtype,
        batch_size=q.shape[0],
        query_heads=q.shape[1],
        key_value_heads=k.shape[1],
        sequence_length=q.shape[2],
        head_dimension=q.shape[-1],
        value_dimension=v.shape[-1],
        gmax=linear.shape[-1],
        causal=True,
        group_layout="per_head" if dim_groups.ndim == 2 else "shared",
        coefficient_layout="per_head",
        stride_class=_stride_class(q, k, v, dim_groups, constant, linear, quadratic),
        requested_gradient_mask=(
            q.requires_grad,
            k.requires_grad,
            v.requires_grad,
            constant.requires_grad,
            linear.requires_grad,
            quadratic.requires_grad,
        ),
        workspace_budget_bytes=workspace_budget_bytes,
    )


def build_forward_kernel_plan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
) -> KernelPlan:
    """Build a geometry-complete plan once for the internal CUDA launch."""
    _validate_triton_inputs(q, k, v, dim_groups, constant, linear, quadratic)
    assert triton is not None
    free_bytes, _ = torch.cuda.mem_get_info(q.device)
    geometry = _runtime_forward_geometry(
        q,
        k,
        v,
        dim_groups,
        constant,
        linear,
        quadratic,
        workspace_budget_bytes=None,
    )
    return build_kernel_plan(geometry, free_bytes=free_bytes)


def _require_live_forward_plan_identity(
    plan: KernelPlan,
    current_geometry: KernelGeometry,
) -> KernelPlan:
    """Require the exact planner result for the live geometry and budget.

    A matching geometry alone is insufficient: the planner version/source,
    workspace contract, specialization axes, runtime axes, and physical-path
    record all participate in the stable ``plan_id`` below.
    """
    if not isinstance(plan, KernelPlan) or not isinstance(
        current_geometry, KernelGeometry
    ):
        raise TypeError(
            "live forward plan identity requires KernelPlan and KernelGeometry"
        )
    if current_geometry.workspace_budget_bytes is None:
        raise ValueError(
            "live forward plan identity requires a resolved workspace budget"
        )
    expected_plan = build_kernel_plan(
        current_geometry,
        workspace_budget_bytes=current_geometry.workspace_budget_bytes,
    )
    if expected_plan.plan_id != plan.plan_id:
        raise ValueError("kernel plan identity does not match the live planner result")
    return expected_plan


def _validate_plan_against_tensors(
    plan: KernelPlan,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
) -> None:
    if not isinstance(plan, KernelPlan):
        raise TypeError("planned grouped causal forward requires a KernelPlan")
    _validate_triton_inputs(q, k, v, dim_groups, constant, linear, quadratic)
    geometry = plan.geometry
    if geometry.workspace_budget_bytes is None:
        raise ValueError("kernel plan must carry a resolved workspace budget")
    current_geometry = _runtime_forward_geometry(
        q,
        k,
        v,
        dim_groups,
        constant,
        linear,
        quadratic,
        workspace_budget_bytes=geometry.workspace_budget_bytes,
    )
    if current_geometry.to_dict() != geometry.to_dict():
        raise ValueError(
            "kernel plan is stale for the live CUDA/tensor/toolchain geometry"
        )
    _require_live_forward_plan_identity(plan, current_geometry)
    free_bytes, _ = torch.cuda.mem_get_info(q.device)
    current_workspace_budget = min(512 * 2**20, free_bytes // 20)
    if current_workspace_budget < geometry.workspace_budget_bytes:
        raise ValueError(
            "live free memory cannot honor the kernel plan workspace budget"
        )
    actual_dtype = str(q.dtype).removeprefix("torch.")
    expected = (
        (geometry.B, q.shape[0]),
        (geometry.Hq, q.shape[1]),
        (geometry.Hkv, k.shape[1]),
        (geometry.N, q.shape[2]),
        (geometry.D, q.shape[-1]),
        (geometry.DV, v.shape[-1]),
        (geometry.gmax, linear.shape[-1]),
    )
    if any(planned != actual for planned, actual in expected):
        raise ValueError("kernel plan geometry does not match actual forward tensors")
    if geometry.dtype != actual_dtype or not geometry.causal:
        raise ValueError("kernel plan dtype/causal mode does not match forward tensors")
    group_layout = "per_head" if dim_groups.ndim == 2 else "shared"
    if (
        geometry.group_layout != group_layout
        or geometry.coefficient_layout != "per_head"
    ):
        raise ValueError(
            "kernel plan group/coefficient layout does not match forward tensors"
        )
    if geometry.stride_class != _stride_class(
        q, k, v, dim_groups, constant, linear, quadratic
    ):
        raise ValueError("kernel plan stride class does not match forward tensors")
    if constant.shape != (geometry.Hq,) or linear.shape != (geometry.Hq, geometry.gmax):
        raise ValueError("kernel plan coefficient shapes do not match forward tensors")
    if quadratic.shape != (geometry.Hq, geometry.gmax, geometry.gmax):
        raise ValueError("kernel plan quadratic shape does not match forward tensors")
    if plan.physical_path != PHYSICAL_PATH_IDENTIFIER:
        raise ValueError("kernel plan does not lock the grouped Hadamard-diag path")
    forward_workspace_allocation_names(plan)


if triton_is_available():

    @triton.jit
    def _grouped_h0_h1_macro_kernel(
        q_pointer,
        k_pointer,
        v_pointer,
        dim_groups_pointer,
        constant_pointer,
        linear_pointer,
        state_pointer,
        numerator_pointer,
        denominator_partial_pointer,
        chunk_start,
        chunk_length,
        query_heads,
        key_value_heads,
        head_dimension,
        value_dimension,
        scale,
        stride_constant_head,
        stride_q_batch,
        stride_q_head,
        stride_q_token,
        stride_q_feature,
        stride_k_batch,
        stride_k_head,
        stride_k_token,
        stride_k_feature,
        stride_v_batch,
        stride_v_head,
        stride_v_token,
        stride_v_value,
        stride_state_batch,
        stride_state_head,
        stride_state_caug,
        stride_state_feature,
        stride_numerator_batch,
        stride_numerator_head,
        stride_numerator_token,
        stride_denominator_partial_batch,
        stride_denominator_partial_head,
        stride_denominator_partial_token,
        GROUPS_PER_HEAD: tl.constexpr,
        GMAX: tl.constexpr,
        D_PAD: tl.constexpr,
        GQA: tl.constexpr,
        BT: tl.constexpr,
        BV: tl.constexpr,
    ):
        value_tile = tl.program_id(0)
        batch_key_value_head = tl.program_id(1)
        batch = batch_key_value_head // key_value_heads
        key_value_head = batch_key_value_head % key_value_heads
        values = value_tile * BV + tl.arange(0, BV)
        features = tl.arange(0, D_PAD)
        feature_mask = features < head_dimension
        logical_value_mask = values < value_dimension + 1
        value_mask = values < value_dimension
        state_base = (
            batch.to(tl.int64) * stride_state_batch
            + key_value_head.to(tl.int64) * stride_state_head
        )
        h0 = tl.load(
            state_pointer + state_base + values * stride_state_caug,
            mask=logical_value_mask,
            other=0.0,
        ).to(tl.float32)
        h1 = tl.load(
            state_pointer
            + state_base
            + values[None, :] * stride_state_caug
            + (1 + features[:, None]) * stride_state_feature,
            mask=feature_mask[:, None] & logical_value_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        for local_token in tl.range(0, BT):
            active = local_token < chunk_length
            token = chunk_start + local_token
            key = tl.load(
                k_pointer
                + batch.to(tl.int64) * stride_k_batch
                + key_value_head.to(tl.int64) * stride_k_head
                + token.to(tl.int64) * stride_k_token
                + features * stride_k_feature,
                mask=active & feature_mask,
                other=0.0,
            ).to(tl.float32)
            loaded_values = tl.load(
                v_pointer
                + batch.to(tl.int64) * stride_v_batch
                + key_value_head.to(tl.int64) * stride_v_head
                + token.to(tl.int64) * stride_v_token
                + values * stride_v_value,
                mask=active & value_mask,
                other=0.0,
            ).to(tl.float32)
            augmented_values = tl.where(
                active & (values == value_dimension), 1.0, loaded_values
            )
            # H0/H1 are KV-owned FP32 Hadamard states and have no A/B/C in
            # their construction; this base term is only a deferred partial.
            h0 += augmented_values
            h1 += key[:, None] * augmented_values[None, :]
            for local_query in range(0, GQA):
                query_head = key_value_head * GQA + local_query
                query = tl.load(
                    q_pointer
                    + batch.to(tl.int64) * stride_q_batch
                    + query_head.to(tl.int64) * stride_q_head
                    + token.to(tl.int64) * stride_q_token
                    + features * stride_q_feature,
                    mask=active & feature_mask,
                    other=0.0,
                ).to(tl.float32)
                if GROUPS_PER_HEAD:
                    groups = tl.load(
                        dim_groups_pointer
                        + query_head.to(tl.int64) * head_dimension
                        + features,
                        mask=feature_mask,
                        other=0,
                    )
                else:
                    groups = tl.load(
                        dim_groups_pointer + features, mask=feature_mask, other=0
                    )
                linear = tl.load(
                    linear_pointer + query_head.to(tl.int64) * GMAX + groups,
                    mask=feature_mask,
                    other=0.0,
                ).to(tl.float32)
                constant = tl.load(
                    constant_pointer + query_head.to(tl.int64) * stride_constant_head,
                    mask=active,
                    other=0.0,
                ).to(tl.float32)
                base_partial = constant * h0 + tl.sum(
                    (query * scale * linear)[:, None] * h1, axis=0
                )
                tl.store(
                    numerator_pointer
                    + batch.to(tl.int64) * stride_numerator_batch
                    + query_head.to(tl.int64) * stride_numerator_head
                    + token.to(tl.int64) * stride_numerator_token
                    + values,
                    base_partial,
                    mask=active & value_mask,
                )
                base_denominator_partial = tl.sum(
                    tl.where(values == value_dimension, base_partial, 0.0), axis=0
                )
                denominator_tile = (value_tile * BV <= value_dimension) & (
                    value_dimension < (value_tile + 1) * BV
                )
                tl.store(
                    denominator_partial_pointer
                    + batch.to(tl.int64) * stride_denominator_partial_batch
                    + query_head.to(tl.int64) * stride_denominator_partial_head
                    + token.to(tl.int64) * stride_denominator_partial_token,
                    base_denominator_partial,
                    mask=active & denominator_tile,
                )

        tl.store(
            state_pointer + state_base + values * stride_state_caug,
            h0,
            mask=logical_value_mask,
        )
        tl.store(
            state_pointer
            + state_base
            + values[None, :] * stride_state_caug
            + (1 + features[:, None]) * stride_state_feature,
            h1,
            mask=feature_mask[:, None] & logical_value_mask[None, :],
        )

    @triton.jit
    def _grouped_h2_all_pair_macro_kernel(
        q_pointer,
        k_pointer,
        v_pointer,
        dim_groups_pointer,
        quadratic_pointer,
        state_pointer,
        pair_partials_pointer,
        chunk_start,
        chunk_length,
        query_heads,
        key_value_heads,
        head_dimension,
        value_dimension,
        scale,
        stride_q_batch,
        stride_q_head,
        stride_q_token,
        stride_q_feature,
        stride_k_batch,
        stride_k_head,
        stride_k_token,
        stride_k_feature,
        stride_v_batch,
        stride_v_head,
        stride_v_token,
        stride_v_value,
        stride_state_batch,
        stride_state_head,
        stride_state_caug,
        stride_state_feature,
        stride_partial_batch,
        stride_partial_head,
        stride_partial_pair_group,
        stride_partial_token,
        stride_partial_caug,
        GROUPS_PER_HEAD: tl.constexpr,
        GMAX: tl.constexpr,
        GQA: tl.constexpr,
        BT: tl.constexpr,
        BP: tl.constexpr,
        BV: tl.constexpr,
        P: tl.constexpr,
    ):
        pair_group = tl.program_id(0)
        value_tile = tl.program_id(1)
        batch_key_value_head = tl.program_id(2)
        batch = batch_key_value_head // key_value_heads
        key_value_head = batch_key_value_head % key_value_heads
        pairs = pair_group * BP + tl.arange(0, BP)
        values = value_tile * BV + tl.arange(0, BV)
        pair_mask = pairs < P
        logical_value_mask = values < value_dimension + 1
        value_mask = values < value_dimension
        rows = ((tl.sqrt(8.0 * pairs.to(tl.float32) + 1.0) - 1.0) * 0.5).to(tl.int32)
        columns = pairs - rows * (rows + 1) // 2
        state_base = (
            batch.to(tl.int64) * stride_state_batch
            + key_value_head.to(tl.int64) * stride_state_head
        )
        for local_token in tl.range(0, BT):
            active = local_token < chunk_length
            token = chunk_start + local_token
            key_rows = tl.load(
                k_pointer
                + batch.to(tl.int64) * stride_k_batch
                + key_value_head.to(tl.int64) * stride_k_head
                + token.to(tl.int64) * stride_k_token
                + rows * stride_k_feature,
                mask=active & pair_mask,
                other=0.0,
            ).to(tl.float32)
            key_columns = tl.load(
                k_pointer
                + batch.to(tl.int64) * stride_k_batch
                + key_value_head.to(tl.int64) * stride_k_head
                + token.to(tl.int64) * stride_k_token
                + columns * stride_k_feature,
                mask=active & pair_mask,
                other=0.0,
            ).to(tl.float32)
            loaded_values = tl.load(
                v_pointer
                + batch.to(tl.int64) * stride_v_batch
                + key_value_head.to(tl.int64) * stride_v_head
                + token.to(tl.int64) * stride_v_token
                + values * stride_v_value,
                mask=active & value_mask,
                other=0.0,
            ).to(tl.float32)
            augmented_values = tl.where(
                active & (values == value_dimension), 1.0, loaded_values
            )
            h2 = tl.load(
                state_pointer
                + state_base
                + values[None, :] * stride_state_caug
                + (1 + head_dimension + pairs[:, None]) * stride_state_feature,
                mask=pair_mask[:, None] & logical_value_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            # Canonical H2 increment is exactly K[r] * K[s] * U; no C/Q/group
            # value participates in the persistent state update.
            h2 += (key_rows * key_columns)[:, None] * augmented_values[None, :]
            tl.store(
                state_pointer
                + state_base
                + values[None, :] * stride_state_caug
                + (1 + head_dimension + pairs[:, None]) * stride_state_feature,
                h2,
                mask=pair_mask[:, None] & logical_value_mask[None, :],
            )
            for local_query in range(0, GQA):
                query_head = key_value_head * GQA + local_query
                query_rows = tl.load(
                    q_pointer
                    + batch.to(tl.int64) * stride_q_batch
                    + query_head.to(tl.int64) * stride_q_head
                    + token.to(tl.int64) * stride_q_token
                    + rows * stride_q_feature,
                    mask=active & pair_mask,
                    other=0.0,
                ).to(tl.float32)
                query_columns = tl.load(
                    q_pointer
                    + batch.to(tl.int64) * stride_q_batch
                    + query_head.to(tl.int64) * stride_q_head
                    + token.to(tl.int64) * stride_q_token
                    + columns * stride_q_feature,
                    mask=active & pair_mask,
                    other=0.0,
                ).to(tl.float32)
                if GROUPS_PER_HEAD:
                    row_groups = tl.load(
                        dim_groups_pointer
                        + query_head.to(tl.int64) * head_dimension
                        + rows,
                        mask=pair_mask,
                        other=0,
                    )
                    column_groups = tl.load(
                        dim_groups_pointer
                        + query_head.to(tl.int64) * head_dimension
                        + columns,
                        mask=pair_mask,
                        other=0,
                    )
                else:
                    row_groups = tl.load(
                        dim_groups_pointer + rows, mask=pair_mask, other=0
                    )
                    column_groups = tl.load(
                        dim_groups_pointer + columns, mask=pair_mask, other=0
                    )
                quadratic = tl.load(
                    quadratic_pointer
                    + query_head.to(tl.int64) * (GMAX * GMAX)
                    + row_groups * GMAX
                    + column_groups,
                    mask=pair_mask,
                    other=0.0,
                ).to(tl.float32)
                multiplicity = tl.where(rows == columns, 1.0, 2.0)
                pair_weight = (
                    multiplicity
                    * quadratic
                    * query_rows
                    * query_columns
                    * scale
                    * scale
                )
                partial = tl.sum(pair_weight[:, None] * h2, axis=0)
                tl.store(
                    pair_partials_pointer
                    + batch.to(tl.int64) * stride_partial_batch
                    + query_head.to(tl.int64) * stride_partial_head
                    + pair_group * stride_partial_pair_group
                    + local_token * stride_partial_token
                    + values * stride_partial_caug,
                    partial,
                    mask=active & logical_value_mask,
                )

    @triton.jit
    def _reduce_all_pair_partials_kernel(
        numerator_pointer,
        denominator_pointer,
        denominator_partial_pointer,
        pair_partials_pointer,
        chunk_start,
        chunk_length,
        query_heads,
        value_dimension,
        stride_numerator_batch,
        stride_numerator_head,
        stride_numerator_token,
        stride_denominator_batch,
        stride_denominator_head,
        stride_denominator_token,
        stride_denominator_partial_batch,
        stride_denominator_partial_head,
        stride_denominator_partial_token,
        stride_partial_batch,
        stride_partial_head,
        stride_partial_pair_group,
        stride_partial_token,
        stride_partial_caug,
        BT: tl.constexpr,
        BV: tl.constexpr,
        PG: tl.constexpr,
    ):
        value_tile = tl.program_id(0)
        batch_query_head = tl.program_id(1)
        batch = batch_query_head // query_heads
        query_head = batch_query_head % query_heads
        tokens = tl.arange(0, BT)
        values = value_tile * BV + tl.arange(0, BV)
        token_mask = tokens < chunk_length
        logical_value_mask = values < value_dimension + 1
        value_mask = values < value_dimension
        partial_sum = tl.zeros((BT, BV), tl.float32)
        # PG is the full all-pair grid cardinality.  This exact ascending loop
        # defines the fixed canonical pair-group reduction order.
        for pair_group in range(0, PG):
            partial_sum += tl.load(
                pair_partials_pointer
                + batch.to(tl.int64) * stride_partial_batch
                + query_head.to(tl.int64) * stride_partial_head
                + pair_group * stride_partial_pair_group
                + tokens[:, None] * stride_partial_token
                + values[None, :] * stride_partial_caug,
                mask=token_mask[:, None] & logical_value_mask[None, :],
                other=0.0,
            ).to(tl.float32)
        base_numerator = tl.load(
            numerator_pointer
            + batch.to(tl.int64) * stride_numerator_batch
            + query_head.to(tl.int64) * stride_numerator_head
            + (chunk_start + tokens[:, None]) * stride_numerator_token
            + values[None, :],
            mask=token_mask[:, None] & value_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        tl.store(
            numerator_pointer
            + batch.to(tl.int64) * stride_numerator_batch
            + query_head.to(tl.int64) * stride_numerator_head
            + (chunk_start + tokens[:, None]) * stride_numerator_token
            + values[None, :],
            base_numerator + partial_sum,
            mask=token_mask[:, None] & value_mask[None, :],
        )
        denominator_partial = tl.sum(
            tl.where(values[None, :] == value_dimension, partial_sum, 0.0), axis=1
        )
        denominator_tile = (value_tile * BV <= value_dimension) & (
            value_dimension < (value_tile + 1) * BV
        )
        base_denominator = tl.load(
            denominator_partial_pointer
            + batch.to(tl.int64) * stride_denominator_partial_batch
            + query_head.to(tl.int64) * stride_denominator_partial_head
            + (chunk_start + tokens) * stride_denominator_partial_token,
            mask=token_mask & denominator_tile,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            denominator_pointer
            + batch.to(tl.int64) * stride_denominator_batch
            + query_head.to(tl.int64) * stride_denominator_head
            + (chunk_start + tokens) * stride_denominator_token,
            base_denominator + denominator_partial,
            mask=token_mask & denominator_tile,
        )

    @triton.jit
    def _normalize_write_kernel(
        numerator_pointer,
        denominator_pointer,
        output_pointer,
        chunk_start,
        chunk_length,
        query_heads,
        value_dimension,
        stride_numerator_batch,
        stride_numerator_head,
        stride_numerator_token,
        stride_denominator_batch,
        stride_denominator_head,
        stride_denominator_token,
        stride_output_batch,
        stride_output_head,
        stride_output_token,
        BT: tl.constexpr,
        BV: tl.constexpr,
    ):
        value_tile = tl.program_id(0)
        batch_query_head = tl.program_id(1)
        batch = batch_query_head // query_heads
        query_head = batch_query_head % query_heads
        tokens = tl.arange(0, BT)
        values = value_tile * BV + tl.arange(0, BV)
        token_mask = tokens < chunk_length
        value_mask = values < value_dimension
        numerator = tl.load(
            numerator_pointer
            + batch.to(tl.int64) * stride_numerator_batch
            + query_head.to(tl.int64) * stride_numerator_head
            + (chunk_start + tokens[:, None]) * stride_numerator_token
            + values[None, :],
            mask=token_mask[:, None] & value_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        denominator = tl.load(
            denominator_pointer
            + batch.to(tl.int64) * stride_denominator_batch
            + query_head.to(tl.int64) * stride_denominator_head
            + (chunk_start + tokens) * stride_denominator_token,
            mask=token_mask,
            other=1.0,
        ).to(tl.float32)
        tl.store(
            output_pointer
            + batch.to(tl.int64) * stride_output_batch
            + query_head.to(tl.int64) * stride_output_head
            + (chunk_start + tokens[:, None]) * stride_output_token
            + values[None, :],
            numerator / denominator[:, None],
            mask=token_mask[:, None] & value_mask[None, :],
        )

    @triton.jit
    def _capture_forward_production_witness_kernel(
        q_pointer,
        k_pointer,
        v_pointer,
        dim_groups_pointer,
        quadratic_pointer,
        state_pointer,
        h0_increment_pointer,
        h1_increment_pointer,
        h2_increment_pointer,
        inclusive_state_pointer,
        diagonal_contraction_pointer,
        batch_index,
        key_value_head,
        query_head,
        token_index,
        pair_index,
        pair_row,
        pair_column,
        channel_index,
        head_dimension,
        value_dimension,
        scale,
        stride_q_batch,
        stride_q_head,
        stride_q_token,
        stride_q_feature,
        stride_k_batch,
        stride_k_head,
        stride_k_token,
        stride_k_feature,
        stride_v_batch,
        stride_v_head,
        stride_v_token,
        stride_v_value,
        stride_state_batch,
        stride_state_head,
        stride_state_caug,
        stride_state_feature,
        GROUPS_PER_HEAD: tl.constexpr,
        GMAX: tl.constexpr,
    ):
        """Read one real final H state and its selected physical contribution."""
        augmented_value = tl.load(
            v_pointer
            + batch_index.to(tl.int64) * stride_v_batch
            + key_value_head.to(tl.int64) * stride_v_head
            + token_index.to(tl.int64) * stride_v_token
            + channel_index * stride_v_value,
        ).to(tl.float32)
        key_row = tl.load(
            k_pointer
            + batch_index.to(tl.int64) * stride_k_batch
            + key_value_head.to(tl.int64) * stride_k_head
            + token_index.to(tl.int64) * stride_k_token
            + pair_row * stride_k_feature,
        ).to(tl.float32)
        key_column = tl.load(
            k_pointer
            + batch_index.to(tl.int64) * stride_k_batch
            + key_value_head.to(tl.int64) * stride_k_head
            + token_index.to(tl.int64) * stride_k_token
            + pair_column * stride_k_feature,
        ).to(tl.float32)
        state_base = (
            batch_index.to(tl.int64) * stride_state_batch
            + key_value_head.to(tl.int64) * stride_state_head
            + channel_index * stride_state_caug
        )
        h0_inclusive = tl.load(state_pointer + state_base).to(tl.float32)
        h1_inclusive = tl.load(
            state_pointer + state_base + (1 + pair_row) * stride_state_feature
        ).to(tl.float32)
        h2_inclusive = tl.load(
            state_pointer
            + state_base
            + (1 + head_dimension + pair_index) * stride_state_feature
        ).to(tl.float32)
        if GROUPS_PER_HEAD:
            row_group = tl.load(
                dim_groups_pointer + query_head.to(tl.int64) * head_dimension + pair_row
            )
            column_group = tl.load(
                dim_groups_pointer
                + query_head.to(tl.int64) * head_dimension
                + pair_column
            )
        else:
            row_group = tl.load(dim_groups_pointer + pair_row)
            column_group = tl.load(dim_groups_pointer + pair_column)
        quadratic = tl.load(
            quadratic_pointer
            + query_head.to(tl.int64) * (GMAX * GMAX)
            + row_group * GMAX
            + column_group,
        ).to(tl.float32)
        query_row = tl.load(
            q_pointer
            + batch_index.to(tl.int64) * stride_q_batch
            + query_head.to(tl.int64) * stride_q_head
            + token_index.to(tl.int64) * stride_q_token
            + pair_row * stride_q_feature,
        ).to(tl.float32)
        query_column = tl.load(
            q_pointer
            + batch_index.to(tl.int64) * stride_q_batch
            + query_head.to(tl.int64) * stride_q_head
            + token_index.to(tl.int64) * stride_q_token
            + pair_column * stride_q_feature,
        ).to(tl.float32)
        tl.store(h0_increment_pointer, augmented_value)
        tl.store(h1_increment_pointer, key_row * augmented_value)
        tl.store(h2_increment_pointer, key_row * key_column * augmented_value)
        tl.store(inclusive_state_pointer, h0_inclusive)
        tl.store(inclusive_state_pointer + 1, h1_inclusive)
        tl.store(inclusive_state_pointer + 2, h2_inclusive)
        # The caller accepts only a diagonal packed pair, whose multiplicity
        # is exactly one.  This reads the actual H2 state produced above.
        tl.store(
            diagonal_contraction_pointer,
            (query_row * scale) * (query_column * scale) * quadratic * h2_inclusive,
        )


def _packed_pair_row_column(pair_index: int) -> tuple[int, int]:
    """Invert canonical lower-triangular packed index ``p(r,s)`` on host."""
    if not isinstance(pair_index, int) or pair_index < 0:
        raise ValueError("production witness pair_index must be non-negative")
    row = 0
    while (row + 1) * (row + 2) // 2 <= pair_index:
        row += 1
    return row, pair_index - row * (row + 1) // 2


def _capture_forward_production_witness(
    *,
    session: ProductionWitnessSession,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    quadratic: torch.Tensor,
    state: torch.Tensor,
    plan: KernelPlan,
    scale: float,
) -> None:
    """Capture deterministic scalar probes from the real final forward state."""
    names = forward_production_witness_capture_names()
    requested = tuple(name for name in names if session.expects(name))
    if not requested:
        return
    if requested != names:
        raise RuntimeError(
            "forward production witness must request every forward probe"
        )
    probe = session.request.probe_for("forward")
    row, column = _packed_pair_row_column(probe.pair_index)
    geometry = plan.geometry
    if (
        probe.batch_index >= geometry.B
        or probe.key_value_head >= geometry.Hkv
        or probe.query_head >= geometry.Hq
        or probe.token_index != geometry.N - 1
        or probe.pair_index >= geometry.pair_count
        or probe.channel_index >= geometry.DV
        or row != column
        or probe.query_head // (geometry.Hq // geometry.Hkv) != probe.key_value_head
    ):
        raise ValueError(
            "forward production witness probe is not a final diagonal GQA coordinate"
        )
    buffers = {
        name: torch.empty(
            (3,) if name == "forward.inclusive_state" else (1,),
            device=q.device,
            dtype=torch.float32,
        )
        for name in names
    }
    assert triton is not None
    _capture_forward_production_witness_kernel[(1,)](
        q,
        k,
        v,
        dim_groups,
        quadratic,
        state,
        *(buffers[name] for name in names),
        probe.batch_index,
        probe.key_value_head,
        probe.query_head,
        probe.token_index,
        probe.pair_index,
        row,
        column,
        probe.channel_index,
        geometry.D,
        geometry.DV,
        scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        state.stride(0),
        state.stride(1),
        state.stride(2),
        state.stride(3),
        GROUPS_PER_HEAD=dim_groups.ndim == 2,
        GMAX=geometry.gmax,
        num_warps=1,
    )
    for name in names:
        session.capture_device_tensor(name, buffers[name])


def grouped_causal_triton_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    *,
    scale: float,
    kernel_plan: KernelPlan | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the internal, plan-bound all-pair macro forward path."""
    plan = kernel_plan or build_forward_kernel_plan(
        q, k, v, dim_groups, constant, linear, quadratic
    )
    _validate_plan_against_tensors(
        plan, q, k, v, dim_groups, constant, linear, quadratic
    )
    assert triton is not None
    global _LAST_FORWARD_PLAN_ID
    _LAST_FORWARD_PLAN_ID = plan.plan_id

    workspace = allocate_planned_forward_workspace(plan, device=q.device)
    state = workspace["forward_state"]
    pair_partials = workspace["forward_pair_partial"]
    output = workspace["output"]
    numerator = workspace["numerator"]
    denominator = workspace["denominator"]
    denominator_partial = workspace["denominator_partial"]
    batch_size, query_heads, _, head_dimension = q.shape
    key_value_heads = k.shape[1]
    value_dimension = v.shape[-1]
    macro_token_block = plan.specialization_axes["macro_token_block"]
    pair_block = plan.specialization_axes["pair_block"]
    value_block = plan.specialization_axes["value_block"]
    if not all(
        isinstance(value, int) and value > 0
        for value in (macro_token_block, pair_block, value_block)
    ):
        raise ValueError("kernel plan has invalid forward specialization axes")
    pair_groups = (plan.geometry.pair_count + pair_block - 1) // pair_block
    if pair_partials.shape != (
        batch_size,
        query_heads,
        pair_groups,
        macro_token_block,
        value_dimension + 1,
    ):
        raise ValueError(
            "kernel plan forward_pair_partial does not match all-pair geometry"
        )
    if denominator_partial.shape != (
        batch_size,
        query_heads,
        plan.geometry.N,
    ):
        raise ValueError(
            "kernel plan denominator_partial does not match token geometry"
        )
    groups_per_head = dim_groups.ndim == 2
    gqa = query_heads // key_value_heads
    augmented_value_tiles = triton.cdiv(value_dimension + 1, value_block)
    output_value_tiles = triton.cdiv(value_dimension, value_block)
    batch_key_value_heads = batch_size * key_value_heads
    batch_query_heads = batch_size * query_heads

    def _launch_macro(*, chunk_start: int, chunk_length: int) -> None:
        _grouped_h0_h1_macro_kernel[(augmented_value_tiles, batch_key_value_heads)](
            q,
            k,
            v,
            dim_groups,
            constant,
            linear,
            state,
            numerator,
            denominator_partial,
            chunk_start,
            chunk_length,
            query_heads,
            key_value_heads,
            head_dimension,
            value_dimension,
            scale,
            constant.stride(0),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            numerator.stride(0),
            numerator.stride(1),
            numerator.stride(2),
            denominator_partial.stride(0),
            denominator_partial.stride(1),
            denominator_partial.stride(2),
            GROUPS_PER_HEAD=groups_per_head,
            GMAX=plan.geometry.gmax,
            D_PAD=triton.next_power_of_2(head_dimension),
            GQA=gqa,
            BT=macro_token_block,
            BV=value_block,
            num_warps=4,
        )
        _grouped_h2_all_pair_macro_kernel[
            (
                pair_groups,
                augmented_value_tiles,
                batch_key_value_heads,
            )
        ](
            q,
            k,
            v,
            dim_groups,
            quadratic,
            state,
            pair_partials,
            chunk_start,
            chunk_length,
            query_heads,
            key_value_heads,
            head_dimension,
            value_dimension,
            scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            pair_partials.stride(0),
            pair_partials.stride(1),
            pair_partials.stride(2),
            pair_partials.stride(3),
            pair_partials.stride(4),
            GROUPS_PER_HEAD=groups_per_head,
            GMAX=plan.geometry.gmax,
            GQA=gqa,
            BT=macro_token_block,
            BP=pair_block,
            BV=value_block,
            P=plan.geometry.pair_count,
            num_warps=4,
        )
        _reduce_all_pair_partials_kernel[(augmented_value_tiles, batch_query_heads)](
            numerator,
            denominator,
            denominator_partial,
            pair_partials,
            chunk_start,
            chunk_length,
            query_heads,
            value_dimension,
            numerator.stride(0),
            numerator.stride(1),
            numerator.stride(2),
            denominator.stride(0),
            denominator.stride(1),
            denominator.stride(2),
            denominator_partial.stride(0),
            denominator_partial.stride(1),
            denominator_partial.stride(2),
            pair_partials.stride(0),
            pair_partials.stride(1),
            pair_partials.stride(2),
            pair_partials.stride(3),
            pair_partials.stride(4),
            BT=macro_token_block,
            BV=value_block,
            PG=pair_groups,
            num_warps=4,
        )
        _normalize_write_kernel[(output_value_tiles, batch_query_heads)](
            numerator,
            denominator,
            output,
            chunk_start,
            chunk_length,
            query_heads,
            value_dimension,
            numerator.stride(0),
            numerator.stride(1),
            numerator.stride(2),
            denominator.stride(0),
            denominator.stride(1),
            denominator.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            BT=macro_token_block,
            BV=value_block,
            num_warps=4,
        )

    launch_planned_forward_macros(plan, _launch_macro)
    session = active_production_witness_session()
    if session is not None:
        _capture_forward_production_witness(
            session=session,
            q=q,
            k=k,
            v=v,
            dim_groups=dim_groups,
            quadratic=quadratic,
            state=state,
            plan=plan,
            scale=scale,
        )
    return output, numerator, denominator


def last_forward_plan_id() -> str | None:
    """Return the internal CUDA plan identifier retained for diagnostics."""
    return _LAST_FORWARD_PLAN_ID


__all__ = (
    "allocate_planned_forward_workspace",
    "build_forward_kernel_plan",
    "forward_production_witness_capture_names",
    "forward_workspace_allocation_names",
    "grouped_causal_triton_forward",
    "last_forward_plan_id",
    "launch_planned_forward_macros",
    "planned_forward_macro_ranges",
    "triton_is_available",
)
