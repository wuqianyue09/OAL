"""KV-owned causal grouped-quadratic backward scan components.

The public autograd boundary remains in ``grouped_quadratic_backward``.  This
module owns the streaming causal VJP pieces so normalization, prefix, and
suffix scheduling can be developed and audited independently.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
from pathlib import Path

import torch

from ..grouped_observability import _observe_grouped_stage
from . import grouped_quadratic_causal_backward_kernels as triton_backward_kernels
from . import grouped_quadratic_causal_prefix_kernels as triton_prefix_kernels
from .grouped_quadratic_causal_common import (
    PHYSICAL_PATH_IDENTIFIER,
    GroupedCausalBackwardDiagnosticWitness,
    GroupedCausalStateSnapshot,
    GroupedCausalSuffixStateSnapshot,
    KernelGeometry,
    KernelPlan,
    NORMALIZATION_VALUE_BLOCK,
    build_kernel_plan,
    canonical_packed_pair_metadata,
    canonical_group_indices,
    collect_cuda_device_properties,
    cuda_device_index,
    require_cpu_diagnostic_geometry,
)
from .grouped_quadratic_causal_forward import grouped_causal_forward_diagnostic_witness
from .grouped_quadratic_production_witness import active_production_witness_session

try:
    import triton
    import triton.language as tl
except (ImportError, ModuleNotFoundError):
    triton = None
    tl = None


_TOKEN_BLOCK = 16
_VALUE_BLOCK = NORMALIZATION_VALUE_BLOCK
_NORMALIZATION_PRODUCTION_WITNESS_CAPTURE_NAMES = ("normalization.g",)


def triton_is_available() -> bool:
    """Return whether the bounded CUDA normalization kernels are importable."""
    return triton is not None and tl is not None


def normalization_production_witness_capture_names() -> tuple[str, ...]:
    """Return the one direct probe of the actual normalization-G tensor."""
    return _NORMALIZATION_PRODUCTION_WITNESS_CAPTURE_NAMES


def _capture_normalization_production_witness(grad_augmented: torch.Tensor) -> None:
    """Digest one real FP32 G entry only during an active evidence session."""
    session = active_production_witness_session()
    if session is None or not session.expects("normalization.g"):
        return
    probe = session.request.probe_for("normalization")
    if (
        probe.batch_index >= grad_augmented.shape[0]
        or probe.query_head >= grad_augmented.shape[1]
        or probe.token_index >= grad_augmented.shape[2]
        or probe.channel_index >= grad_augmented.shape[3]
    ):
        raise ValueError("normalization production witness probe is out of bounds")
    # ``grad_augmented`` is the FP32 CUDA output of the actual normalization
    # kernels.  A one-element view is contiguous and needs no GPU copy.
    session.capture_device_tensor(
        "normalization.g",
        grad_augmented[
            probe.batch_index,
            probe.query_head,
            probe.token_index,
            probe.channel_index,
        ].reshape(1),
    )


def _backward_stage_from_mask(
    requested_mask: tuple[bool, bool, bool, bool, bool, bool],
) -> str:
    """Classify the one physical VJP stage selected by its exact output mask."""
    if _requires_prefix_vjp(requested_mask) and not any(requested_mask[1:3]):
        return "prefix"
    if (requested_mask[1] or requested_mask[2]) and not (
        requested_mask[0] or any(requested_mask[3:])
    ):
        return "suffix"
    raise ValueError(
        "a planned causal VJP stage must be either Q/A/B/C prefix or K/V suffix"
    )


@lru_cache(maxsize=2)
def _backward_stage_source_hash(stage: str) -> str:
    """Bind each plan only to the kernel source it can actually launch."""
    if stage == "prefix":
        stage_source = Path(triton_prefix_kernels.__file__).read_bytes()
    elif stage == "suffix":
        stage_source = Path(triton_backward_kernels.__file__).read_bytes()
    else:
        raise ValueError("backward stage must be prefix or suffix")
    return hashlib.sha256(Path(__file__).read_bytes() + stage_source).hexdigest()


@lru_cache(maxsize=1)
def _normalization_stage_source_hash() -> str:
    """Bind a plan to the real augmented-normalization Triton entry.

    The two normalization kernels and their launch/shape checks are defined in
    this module.  Keeping this hash separate from the H-prefix/suffix hashes
    makes evidence identify the actual G-producing stage without changing the
    H0/H1/H2 Hadamard-diagonal recurrence.
    """
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _prefix_stride_class_from_metadata(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    *,
    grad_augmented_is_contiguous: bool,
) -> str:
    """Classify the fixed G layout without allocating a synthetic G tensor."""
    if (
        not all(
            tensor.is_contiguous()
            for tensor in (q, k, v, dim_groups, linear, quadratic)
        )
        or not grad_augmented_is_contiguous
    ):
        raise ValueError(
            "planned grouped causal prefix requires contiguous Q/K/V/G/groups/B/C"
        )
    if constant.ndim != 1 or constant.stride(0) <= 0:
        raise ValueError(
            "planned grouped causal prefix requires a positive-stride [Hq] A"
        )
    return "contiguous" if constant.is_contiguous() else "constant_head_strided"


def _validate_requested_gradient_mask(
    needs_input_grad: tuple[bool, bool, bool, bool, bool, bool],
) -> tuple[bool, bool, bool, bool, bool, bool]:
    normalized = tuple(needs_input_grad)
    if len(normalized) != 6 or not all(isinstance(value, bool) for value in normalized):
        raise TypeError("needs_input_grad must be six booleans for Q/K/V/A/B/C")
    return normalized


def _validate_backward_plan_metadata_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
) -> None:
    """Validate all non-G inputs needed to plan the exact causal VJP stages."""
    if not all(
        isinstance(tensor, torch.Tensor)
        for tensor in (q, k, v, dim_groups, constant, linear, quadratic)
    ):
        raise TypeError("grouped causal VJP plan inputs must be tensors")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Q/K/V must be rank-4 tensors")
    batch_size, query_heads, token_count, head_dimension = q.shape
    if k.shape[0] != batch_size or v.shape[0] != batch_size:
        raise ValueError("Q/K/V batch dimensions must agree")
    if k.shape[2] != token_count or v.shape[2] != token_count:
        raise ValueError("Q/K/V token dimensions must agree")
    if k.shape[-1] != head_dimension:
        raise ValueError("Q and K head dimensions must agree")
    if query_heads % k.shape[1]:
        raise ValueError("query heads must be divisible by KV heads")
    gmax = linear.shape[-1] if linear.ndim == 2 else 0
    if (
        constant.shape != (query_heads,)
        or linear.shape != (query_heads, gmax)
        or quadratic.shape != (query_heads, gmax, gmax)
    ):
        raise ValueError("expanded A/B/C shapes must agree with Q heads and Gmax")
    if not all(
        tensor.device == q.device
        for tensor in (k, v, dim_groups, constant, linear, quadratic)
    ):
        raise ValueError("grouped causal VJP plan tensors must share one device")
    _canonical_groups(
        dim_groups,
        query_heads=query_heads,
        head_dimension=head_dimension,
        gmax=gmax,
    )


def _runtime_cuda_hardware_properties(
    q: torch.Tensor,
) -> tuple[object, dict[str, int]]:
    """Collect the one live dynamic hardware record shared by every VJP stage."""
    properties = torch.cuda.get_device_properties(q.device)
    return properties, collect_cuda_device_properties(
        properties,
        device_index=cuda_device_index(q.device),
        triton_module=triton,
    )


def _runtime_backward_prefix_geometry_from_metadata(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    *,
    needs_input_grad: tuple[bool, bool, bool, bool, bool, bool],
    workspace_budget_bytes: int | None,
    grad_augmented_is_contiguous: bool,
) -> KernelGeometry:
    """Describe a VJP stage without materializing a speculative G buffer."""
    requested_mask = _validate_requested_gradient_mask(needs_input_grad)
    _validate_backward_plan_metadata_inputs(
        q, k, v, dim_groups, constant, linear, quadratic
    )
    assert triton is not None
    properties, hardware = _runtime_cuda_hardware_properties(q)
    return KernelGeometry(
        compute_capability=(properties.major, properties.minor),
        **hardware,
        torch_version=torch.__version__,
        triton_version=str(getattr(triton, "__version__", "unknown")),
        source_hash=_backward_stage_source_hash(
            _backward_stage_from_mask(requested_mask)
        ),
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
        stride_class=_prefix_stride_class_from_metadata(
            q,
            k,
            v,
            dim_groups,
            constant,
            linear,
            quadratic,
            grad_augmented_is_contiguous=grad_augmented_is_contiguous,
        ),
        requested_gradient_mask=requested_mask,
        workspace_budget_bytes=workspace_budget_bytes,
    )


def _runtime_normalization_geometry_from_metadata(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    *,
    needs_input_grad: tuple[bool, bool, bool, bool, bool, bool],
    workspace_budget_bytes: int | None,
) -> KernelGeometry:
    """Describe the actual FP32 augmented-normalization VJP launch.

    It retains the full public Q/K/V/A/B/C mask as an admission discriminator,
    but its source and workspace are dedicated to G construction rather than
    the separate KV-owned H-prefix/suffix scans.
    """
    requested_mask = _validate_requested_gradient_mask(needs_input_grad)
    if not any(requested_mask):
        raise ValueError(
            "normalization VJP requires at least one requested input gradient"
        )
    _validate_backward_plan_metadata_inputs(
        q, k, v, dim_groups, constant, linear, quadratic
    )
    assert triton is not None
    properties, hardware = _runtime_cuda_hardware_properties(q)
    return KernelGeometry(
        compute_capability=(properties.major, properties.minor),
        **hardware,
        torch_version=torch.__version__,
        triton_version=str(getattr(triton, "__version__", "unknown")),
        source_hash=_normalization_stage_source_hash(),
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
        stride_class=_prefix_stride_class_from_metadata(
            q,
            k,
            v,
            dim_groups,
            constant,
            linear,
            quadratic,
            grad_augmented_is_contiguous=True,
        ),
        requested_gradient_mask=requested_mask,
        workspace_budget_bytes=workspace_budget_bytes,
        launch_stage="normalization_vjp",
    )


def _runtime_backward_prefix_geometry(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    grad_augmented: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    *,
    needs_input_grad: tuple[bool, bool, bool, bool, bool, bool],
    workspace_budget_bytes: int | None,
) -> KernelGeometry:
    """Describe the complete live CUDA prefix-VJP discriminator tuple."""
    requested_mask = _validate_requested_gradient_mask(needs_input_grad)
    triton_prefix_kernels._validate_triton_inputs(
        q,
        k,
        v,
        dim_groups,
        grad_augmented,
        constant,
        linear,
        quadratic,
    )
    _validate_grouped_scan_inputs(
        q,
        k,
        v,
        dim_groups,
        grad_augmented,
        constant,
        linear,
        quadratic,
        scale=1.0,
    )
    return _runtime_backward_prefix_geometry_from_metadata(
        q,
        k,
        v,
        dim_groups,
        constant,
        linear,
        quadratic,
        needs_input_grad=requested_mask,
        workspace_budget_bytes=workspace_budget_bytes,
        grad_augmented_is_contiguous=grad_augmented.is_contiguous(),
    )


def _require_live_backward_prefix_plan_identity(
    plan: KernelPlan,
    current_geometry: KernelGeometry,
) -> KernelPlan:
    """Require the exact planner payload for the live prefix geometry/budget."""
    if not isinstance(plan, KernelPlan) or not isinstance(
        current_geometry, KernelGeometry
    ):
        raise TypeError(
            "live backward prefix plan identity requires KernelPlan and KernelGeometry"
        )
    if current_geometry.workspace_budget_bytes is None:
        raise ValueError(
            "live backward prefix plan identity requires a resolved workspace budget"
        )
    expected_plan = build_kernel_plan(
        current_geometry,
        workspace_budget_bytes=current_geometry.workspace_budget_bytes,
    )
    if expected_plan.plan_id != plan.plan_id:
        raise ValueError(
            "backward prefix plan identity does not match the live planner result"
        )
    return expected_plan


def build_backward_kernel_plan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    grad_augmented: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    *,
    needs_input_grad: tuple[bool, bool, bool, bool, bool, bool],
) -> KernelPlan:
    """Build the exact CUDA H-prefix/suffix workspace plan for one VJP mask."""
    requested_mask = _validate_requested_gradient_mask(needs_input_grad)
    geometry = _runtime_backward_prefix_geometry(
        q,
        k,
        v,
        dim_groups,
        grad_augmented,
        constant,
        linear,
        quadratic,
        needs_input_grad=requested_mask,
        workspace_budget_bytes=None,
    )
    free_bytes, _ = torch.cuda.mem_get_info(q.device)
    return build_kernel_plan(geometry, free_bytes=free_bytes)


def build_backward_kernel_plan_from_metadata(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    *,
    needs_input_grad: tuple[bool, bool, bool, bool, bool, bool],
) -> KernelPlan:
    """Plan a contiguous future G without allocating one before admission.

    ``augmented_normalization_vjp`` always produces the contiguous FP32
    ``[B,Hq,N,DV+1]`` G consumed by the planned CUDA VJP.  This metadata-only
    construction therefore has the same geometry/stride discriminator as the
    later real G while allocating neither a speculative G nor a second
    workspace buffer.
    """
    requested_mask = _validate_requested_gradient_mask(needs_input_grad)
    if not q.is_cuda:
        raise RuntimeError("grouped causal VJP plan metadata requires CUDA tensors")
    geometry = _runtime_backward_prefix_geometry_from_metadata(
        q,
        k,
        v,
        dim_groups,
        constant,
        linear,
        quadratic,
        needs_input_grad=requested_mask,
        workspace_budget_bytes=None,
        grad_augmented_is_contiguous=True,
    )
    free_bytes, _ = torch.cuda.mem_get_info(q.device)
    return build_kernel_plan(geometry, free_bytes=free_bytes)


def build_normalization_kernel_plan_for_geometry(
    geometry: KernelGeometry,
) -> KernelPlan:
    """Build a pure, source-bound normalization VJP plan for evidence checks."""
    if not isinstance(geometry, KernelGeometry):
        raise TypeError("normalization plan requires a KernelGeometry")
    if not any(geometry.requested_gradient_mask):
        raise ValueError("normalization plan requires a non-empty Q/K/V/A/B/C mask")
    normalized_geometry = replace(
        geometry,
        source_hash=_normalization_stage_source_hash(),
        launch_stage="normalization_vjp",
    )
    return build_kernel_plan(
        normalized_geometry,
        workspace_budget_bytes=normalized_geometry.workspace_budget_bytes,
    )


def build_normalization_kernel_plan_from_metadata(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    *,
    needs_input_grad: tuple[bool, bool, bool, bool, bool, bool],
) -> KernelPlan:
    """Plan the real G-producing Triton normalization stage before admission."""
    requested_mask = _validate_requested_gradient_mask(needs_input_grad)
    geometry = _runtime_normalization_geometry_from_metadata(
        q,
        k,
        v,
        dim_groups,
        constant,
        linear,
        quadratic,
        needs_input_grad=requested_mask,
        workspace_budget_bytes=None,
    )
    free_bytes, _ = torch.cuda.mem_get_info(q.device)
    return build_kernel_plan(geometry, free_bytes=free_bytes)


@dataclass(frozen=True)
class GroupedCausalVjpKernelPlans:
    """The exact prefix/suffix plans admitted for one Q/K/V/A/B/C request."""

    requested_gradient_mask: tuple[bool, bool, bool, bool, bool, bool]
    normalization_plan: KernelPlan | None
    prefix_plan: KernelPlan | None
    suffix_plan: KernelPlan | None

    def __post_init__(self) -> None:
        requested_mask = _validate_requested_gradient_mask(self.requested_gradient_mask)
        prefix_mask = (requested_mask[0], False, False, *requested_mask[3:])
        suffix_mask = (False, requested_mask[1], requested_mask[2], False, False, False)
        prefix_required = _requires_prefix_vjp(requested_mask)
        suffix_required = requested_mask[1] or requested_mask[2]
        normalization_required = any(requested_mask)
        if (self.normalization_plan is None) == normalization_required:
            raise ValueError(
                "VJP normalization plan presence must match its Q/K/V/A/B/C mask"
            )
        if (self.prefix_plan is None) == prefix_required:
            raise ValueError("VJP prefix plan presence must match its Q/A/B/C mask")
        if (self.suffix_plan is None) == suffix_required:
            raise ValueError("VJP suffix plan presence must match its K/V mask")
        for name, plan, expected_mask in (
            ("normalization", self.normalization_plan, requested_mask),
            ("prefix", self.prefix_plan, prefix_mask),
            ("suffix", self.suffix_plan, suffix_mask),
        ):
            if plan is not None:
                if not isinstance(plan, KernelPlan):
                    raise TypeError(f"VJP {name} plan must be a KernelPlan")
                if plan.geometry.requested_gradient_mask != expected_mask:
                    raise ValueError(
                        f"VJP {name} plan has a different Q/K/V/A/B/C gradient mask"
                    )
                expected_source = (
                    _normalization_stage_source_hash()
                    if name == "normalization"
                    else _backward_stage_source_hash(name)
                )
                if plan.geometry.source_hash != expected_source:
                    raise ValueError(
                        f"VJP {name} plan source hash does not match its launch stage"
                    )
                if (
                    name == "normalization"
                    and plan.geometry.launch_stage != "normalization_vjp"
                ):
                    raise ValueError(
                        "VJP normalization plan must carry the normalization launch stage"
                    )
        object.__setattr__(self, "requested_gradient_mask", requested_mask)


def build_causal_vjp_kernel_plans_from_metadata(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    *,
    needs_input_grad: tuple[bool, bool, bool, bool, bool, bool],
) -> GroupedCausalVjpKernelPlans:
    """Construct the exact two VJP stage plans before public admission."""
    requested_mask = _validate_requested_gradient_mask(needs_input_grad)
    prefix_mask = (requested_mask[0], False, False, *requested_mask[3:])
    suffix_mask = (False, requested_mask[1], requested_mask[2], False, False, False)
    normalization_plan = (
        build_normalization_kernel_plan_from_metadata(
            q,
            k,
            v,
            dim_groups,
            constant,
            linear,
            quadratic,
            needs_input_grad=requested_mask,
        )
        if any(requested_mask)
        else None
    )
    prefix_plan = (
        build_backward_kernel_plan_from_metadata(
            q,
            k,
            v,
            dim_groups,
            constant,
            linear,
            quadratic,
            needs_input_grad=prefix_mask,
        )
        if _requires_prefix_vjp(requested_mask)
        else None
    )
    suffix_plan = (
        build_backward_kernel_plan_from_metadata(
            q,
            k,
            v,
            dim_groups,
            constant,
            linear,
            quadratic,
            needs_input_grad=suffix_mask,
        )
        if requested_mask[1] or requested_mask[2]
        else None
    )
    return GroupedCausalVjpKernelPlans(
        requested_gradient_mask=requested_mask,
        normalization_plan=normalization_plan,
        prefix_plan=prefix_plan,
        suffix_plan=suffix_plan,
    )


if triton_is_available():

    @triton.jit
    def _augmented_normalization_partial_kernel(
        grad_output_pointer,
        numerator_pointer,
        denominator_pointer,
        grad_numerator_pointer,
        augmented_pointer,
        denominator_partials_pointer,
        token_count,
        query_heads,
        value_dimension,
        VALUE_BLOCKS: tl.constexpr,
        HAS_GRAD_OUTPUT: tl.constexpr,
        HAS_GRAD_NUMERATOR: tl.constexpr,
        BQ: tl.constexpr,
        BV: tl.constexpr,
    ):
        value_tile = tl.program_id(0)
        token_tile = tl.program_id(1)
        batch_query_head = tl.program_id(2).to(tl.int64)
        token_offsets = token_tile * BQ + tl.arange(0, BQ)
        token_offsets_64 = token_offsets.to(tl.int64)
        value_offsets = value_tile * BV + tl.arange(0, BV)
        token_mask = token_offsets < token_count
        value_mask = value_offsets < value_dimension
        flattened_token_offsets = batch_query_head * token_count + token_offsets_64
        numerator_offsets = (
            flattened_token_offsets[:, None] * value_dimension + value_offsets[None, :]
        )
        augmented_offsets = (
            flattened_token_offsets[:, None] * (value_dimension + 1)
            + value_offsets[None, :]
        )
        denominator = tl.load(
            denominator_pointer + flattened_token_offsets,
            mask=token_mask,
            other=1.0,
        ).to(tl.float32)
        numerator = tl.load(
            numerator_pointer + numerator_offsets,
            mask=token_mask[:, None] & value_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        if HAS_GRAD_OUTPUT:
            grad_output = tl.load(
                grad_output_pointer + numerator_offsets,
                mask=token_mask[:, None] & value_mask[None, :],
                other=0.0,
            ).to(tl.float32)
        else:
            grad_output = tl.zeros((BQ, BV), dtype=tl.float32)
        numerator_gradient = grad_output / denominator[:, None]
        if HAS_GRAD_NUMERATOR:
            numerator_gradient += tl.load(
                grad_numerator_pointer + numerator_offsets,
                mask=token_mask[:, None] & value_mask[None, :],
                other=0.0,
            ).to(tl.float32)
        tl.store(
            augmented_pointer + augmented_offsets,
            numerator_gradient,
            mask=token_mask[:, None] & value_mask[None, :],
        )
        tl.store(
            denominator_partials_pointer
            + flattened_token_offsets * VALUE_BLOCKS
            + value_tile,
            tl.sum(grad_output * numerator, axis=1),
            mask=token_mask,
        )

    @triton.jit
    def _augmented_normalization_reduce_kernel(
        denominator_pointer,
        grad_denominator_pointer,
        augmented_pointer,
        denominator_partials_pointer,
        token_count,
        query_heads,
        value_dimension,
        VALUE_BLOCKS: tl.constexpr,
        HAS_GRAD_DENOMINATOR: tl.constexpr,
        BQ: tl.constexpr,
    ):
        token_tile = tl.program_id(0)
        batch_query_head = tl.program_id(1).to(tl.int64)
        token_offsets = token_tile * BQ + tl.arange(0, BQ)
        token_offsets_64 = token_offsets.to(tl.int64)
        token_mask = token_offsets < token_count
        flattened_token_offsets = batch_query_head * token_count + token_offsets_64
        partial_sum = tl.zeros((BQ,), dtype=tl.float32)
        for value_tile in range(0, VALUE_BLOCKS):
            partial_sum += tl.load(
                denominator_partials_pointer
                + flattened_token_offsets * VALUE_BLOCKS
                + value_tile,
                mask=token_mask,
                other=0.0,
            )
        denominator = tl.load(
            denominator_pointer + flattened_token_offsets,
            mask=token_mask,
            other=1.0,
        ).to(tl.float32)
        denominator_gradient = -partial_sum / (denominator * denominator)
        if HAS_GRAD_DENOMINATOR:
            denominator_gradient += tl.load(
                grad_denominator_pointer + flattened_token_offsets,
                mask=token_mask,
                other=0.0,
            ).to(tl.float32)
        tl.store(
            augmented_pointer
            + flattened_token_offsets * (value_dimension + 1)
            + value_dimension,
            denominator_gradient,
            mask=token_mask,
        )


def _require_gradient(
    name: str,
    gradient: torch.Tensor | None,
    *,
    reference: torch.Tensor,
) -> torch.Tensor | None:
    if gradient is None:
        return None
    if not isinstance(gradient, torch.Tensor):
        raise TypeError(f"{name} must be a tensor or None")
    if gradient.shape != reference.shape:
        raise ValueError(f"{name} must match the corresponding forward output shape")
    if gradient.device != reference.device:
        raise ValueError(f"{name} must be on the same device as the forward output")
    if not gradient.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    return gradient


def _validate_normalization_outputs(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> None:
    if not isinstance(numerator, torch.Tensor) or not isinstance(
        denominator, torch.Tensor
    ):
        raise TypeError("numerator and denominator must be tensors")
    if numerator.ndim != 4 or denominator.ndim != 4:
        raise ValueError("numerator and denominator must be rank-4 tensors")
    if numerator.shape[:-1] != denominator.shape[:-1] or denominator.shape[-1] != 1:
        raise ValueError("denominator must have shape numerator.shape[:-1] + (1,)")
    if numerator.device != denominator.device:
        raise ValueError("numerator and denominator must share one device")
    if not numerator.is_floating_point() or not denominator.is_floating_point():
        raise TypeError("numerator and denominator must have floating-point dtypes")


def _require_live_normalization_plan_identity(
    plan: KernelPlan,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> KernelPlan:
    """Bind a G launch to its source-specific reviewed workspace plan."""
    if not isinstance(plan, KernelPlan):
        raise TypeError("normalization VJP requires a KernelPlan")
    geometry = plan.geometry
    expected_shape = (
        geometry.B,
        geometry.Hq,
        geometry.N,
        geometry.DV,
    )
    if numerator.shape != expected_shape or denominator.shape != (
        *expected_shape[:-1],
        1,
    ):
        raise ValueError(
            "normalization plan geometry does not match live numerator/denominator"
        )
    if (
        geometry.launch_stage != "normalization_vjp"
        or geometry.source_hash != _normalization_stage_source_hash()
        or not any(geometry.requested_gradient_mask)
    ):
        raise ValueError(
            "normalization plan is not bound to the live normalization VJP"
        )
    expected_names = {
        "normalization_numerator",
        "normalization_denominator",
        "normalization_grad_output",
        "normalization_grad_numerator",
        "normalization_grad_denominator",
        "normalization_g",
        "normalization_denominator_partials",
    }
    if {allocation.name for allocation in plan.workspace.allocations} != expected_names:
        raise ValueError("normalization plan does not own the exact VJP workspace")
    expected_plan = build_normalization_kernel_plan_for_geometry(geometry)
    if expected_plan.plan_id != plan.plan_id:
        raise ValueError(
            "normalization plan identity does not match the live planner result"
        )
    return expected_plan


def _validate_backward_diagnostic_forward_outputs(
    q: torch.Tensor,
    v: torch.Tensor,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> None:
    """Validate exact Q-derived forward shapes before building any witness state."""
    expected_numerator_shape = (*q.shape[:-1], v.shape[-1])
    expected_denominator_shape = (*q.shape[:-1], 1)
    for name, tensor, expected_shape in (
        ("numerator", numerator, expected_numerator_shape),
        ("denominator", denominator, expected_denominator_shape),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.shape != expected_shape:
            raise ValueError(f"{name} must have exact Q-derived shape {expected_shape}")
        if tensor.device != q.device:
            raise ValueError(f"{name} must share the Q device")
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype")


def _torch_augmented_normalization_vjp(
    grad_output: torch.Tensor | None,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    grad_numerator: torch.Tensor | None,
    grad_denominator: torch.Tensor | None,
) -> torch.Tensor:
    """CPU fallback and reference-compatible implementation for the VJP."""
    value_dimension = numerator.shape[-1]
    augmented = torch.empty(
        (*numerator.shape[:-1], value_dimension + 1),
        device=numerator.device,
        dtype=torch.float32,
    )
    value_gradient = augmented[..., :value_dimension]
    denominator_gradient = augmented[..., value_dimension:]
    if grad_output is None:
        value_gradient.zero_()
        denominator_gradient.zero_()
    else:
        value_gradient.copy_(grad_output.float()).div_(denominator.float())
        torch.sum(
            grad_output.float() * numerator.float(),
            dim=-1,
            keepdim=True,
            out=denominator_gradient,
        )
        denominator_gradient.div_(denominator.float().square()).neg_()
    if grad_numerator is not None:
        value_gradient.add_(grad_numerator.float())
    if grad_denominator is not None:
        denominator_gradient.add_(grad_denominator.float())
    return augmented


def _triton_augmented_normalization_vjp(
    grad_output: torch.Tensor | None,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    grad_numerator: torch.Tensor | None,
    grad_denominator: torch.Tensor | None,
) -> torch.Tensor:
    """Write G through fixed value-channel partial and reduction kernels."""
    assert triton is not None
    batch_size, query_heads, token_count, value_dimension = numerator.shape
    value_blocks = triton.cdiv(value_dimension, _VALUE_BLOCK)
    numerator_flat = (
        numerator.float().contiguous().reshape(-1, token_count, value_dimension)
    )
    denominator_flat = denominator.float().contiguous().reshape(-1, token_count)
    output_flat = (
        numerator_flat
        if grad_output is None
        else grad_output.float().contiguous().reshape_as(numerator_flat)
    )
    numerator_direct_flat = (
        numerator_flat
        if grad_numerator is None
        else grad_numerator.float().contiguous().reshape_as(numerator_flat)
    )
    denominator_direct_flat = (
        denominator_flat
        if grad_denominator is None
        else grad_denominator.float().contiguous().reshape_as(denominator_flat)
    )
    augmented_flat = torch.empty(
        (batch_size * query_heads, token_count, value_dimension + 1),
        device=numerator.device,
        dtype=torch.float32,
    )
    denominator_partials = torch.empty(
        (batch_size * query_heads, token_count, value_blocks),
        device=numerator.device,
        dtype=torch.float32,
    )
    _augmented_normalization_partial_kernel[
        (value_blocks, triton.cdiv(token_count, _TOKEN_BLOCK), batch_size * query_heads)
    ](
        output_flat,
        numerator_flat,
        denominator_flat,
        numerator_direct_flat,
        augmented_flat,
        denominator_partials,
        token_count,
        query_heads,
        value_dimension,
        VALUE_BLOCKS=value_blocks,
        HAS_GRAD_OUTPUT=grad_output is not None,
        HAS_GRAD_NUMERATOR=grad_numerator is not None,
        BQ=_TOKEN_BLOCK,
        BV=_VALUE_BLOCK,
        num_warps=4,
    )
    _augmented_normalization_reduce_kernel[
        (triton.cdiv(token_count, _TOKEN_BLOCK), batch_size * query_heads)
    ](
        denominator_flat,
        denominator_direct_flat,
        augmented_flat,
        denominator_partials,
        token_count,
        query_heads,
        value_dimension,
        VALUE_BLOCKS=value_blocks,
        HAS_GRAD_DENOMINATOR=grad_denominator is not None,
        BQ=_TOKEN_BLOCK,
        num_warps=4,
    )
    return augmented_flat.reshape(
        batch_size, query_heads, token_count, value_dimension + 1
    )


def augmented_normalization_vjp(
    grad_output: torch.Tensor | None,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    *,
    grad_numerator: torch.Tensor | None = None,
    grad_denominator: torch.Tensor | None = None,
    kernel_plan: KernelPlan | None = None,
) -> torch.Tensor:
    """Combine output/direct gradients into FP32 augmented state channels.

    The last channel corresponds to the ``U=[V,1]`` denominator channel.  It
    is written directly into the final contiguous ``G`` tensor, avoiding the
    legacy split value/count buffers and their later concatenation.
    """
    _validate_normalization_outputs(numerator, denominator)
    if kernel_plan is not None:
        _require_live_normalization_plan_identity(kernel_plan, numerator, denominator)
    checked_output = _require_gradient("grad_output", grad_output, reference=numerator)
    checked_numerator = _require_gradient(
        "grad_numerator", grad_numerator, reference=numerator
    )
    checked_denominator = _require_gradient(
        "grad_denominator", grad_denominator, reference=denominator
    )

    if numerator.is_cuda and triton_is_available():
        return _triton_augmented_normalization_vjp(
            checked_output,
            numerator,
            denominator,
            checked_numerator,
            checked_denominator,
        )
    return _torch_augmented_normalization_vjp(
        checked_output,
        numerator,
        denominator,
        checked_numerator,
        checked_denominator,
    )


def _canonical_groups(
    dim_groups: torch.Tensor,
    *,
    query_heads: int,
    head_dimension: int,
    gmax: int,
) -> torch.Tensor:
    if dim_groups.dtype != torch.int32:
        raise TypeError("dim_groups must be canonical int32")
    if dim_groups.ndim == 1:
        if dim_groups.shape != (head_dimension,):
            raise ValueError("shared dim_groups must have shape [D]")
        groups = dim_groups.unsqueeze(0).expand(query_heads, -1)
    elif dim_groups.ndim == 2:
        if dim_groups.shape != (query_heads, head_dimension):
            raise ValueError("per-head dim_groups must have shape [Hq, D]")
        groups = dim_groups
    else:
        raise ValueError("dim_groups must have shape [D] or [Hq, D]")
    groups = groups.to(dtype=torch.int64)
    if bool(torch.any(groups < 0)) or bool(torch.any(groups >= gmax)):
        raise ValueError("dim_groups must index the expanded coefficient groups")
    return groups


def _validate_grouped_scan_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    grad_augmented: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    *,
    scale: float,
) -> tuple[torch.Tensor, int, int, int, int]:
    if not all(
        isinstance(tensor, torch.Tensor)
        for tensor in (q, k, v, dim_groups, grad_augmented, constant, linear, quadratic)
    ):
        raise TypeError("grouped causal scan inputs must be tensors")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or grad_augmented.ndim != 4:
        raise ValueError("Q/K/V/G must be rank-4 tensors")
    batch_size, query_heads, token_count, head_dimension = q.shape
    if k.shape[0] != batch_size or v.shape[0] != batch_size:
        raise ValueError("Q/K/V batch dimensions must agree")
    if k.shape[2] != token_count or v.shape[2] != token_count:
        raise ValueError("Q/K/V token dimensions must agree")
    if k.shape[-1] != head_dimension:
        raise ValueError("Q and K head dimensions must agree")
    key_value_heads = k.shape[1]
    value_dimension = v.shape[-1]
    if query_heads % key_value_heads != 0:
        raise ValueError("query heads must be divisible by KV heads")
    if grad_augmented.shape != (
        batch_size,
        query_heads,
        token_count,
        value_dimension + 1,
    ):
        raise ValueError("G must have shape [B, Hq, N, DV + 1]")
    gmax = linear.shape[-1]
    if (
        constant.shape != (query_heads,)
        or linear.shape != (query_heads, gmax)
        or quadratic.shape != (query_heads, gmax, gmax)
    ):
        raise ValueError("expanded A/B/C shapes must agree with Q heads and Gmax")
    if not all(
        tensor.device == q.device
        for tensor in (k, v, dim_groups, grad_augmented, constant, linear, quadratic)
    ):
        raise ValueError("grouped causal scan tensors must share one device")
    if not isinstance(scale, (int, float)) or isinstance(scale, bool):
        raise TypeError("scale must be a scalar")
    groups = _canonical_groups(
        dim_groups,
        query_heads=query_heads,
        head_dimension=head_dimension,
        gmax=gmax,
    )
    return groups, batch_size, key_value_heads, value_dimension, head_dimension


def _torch_launch_prefix_dq_dcoeff(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    grad_augmented: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    *,
    scale: float,
    need_q: bool,
    need_coefficients: bool,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Run the exact KV-owned inclusive prefix scan for dQ/dA/dB/dC.

    This correctness-first schedule retains only current H0/H1/H2 state.  It
    deliberately reduces group coefficients in a fixed Python-visible order;
    token and pair histories are never stored.
    """
    if not need_q and not need_coefficients:
        return None, None, None, None
    groups, batch_size, key_value_heads, value_dimension, head_dimension = (
        _validate_grouped_scan_inputs(
            q,
            k,
            v,
            dim_groups,
            grad_augmented,
            constant,
            linear,
            quadratic,
            scale=scale,
        )
    )
    query_heads = q.shape[1]
    token_count = q.shape[2]
    pair_rows, pair_columns = torch.tril_indices(
        head_dimension, head_dimension, device=q.device, dtype=torch.int64
    )
    pair_count = pair_rows.numel()
    multiplicity = torch.where(
        pair_rows == pair_columns,
        torch.ones(pair_count, device=q.device, dtype=torch.float32),
        torch.full((pair_count,), 2.0, device=q.device, dtype=torch.float32),
    )
    query_to_key_value = torch.arange(query_heads, device=q.device) // (
        query_heads // key_value_heads
    )
    q_work = q.float()
    k_work = k.float()
    v_work = v.float()
    g_work = grad_augmented.float()
    constant_work = constant.float()
    linear_work = linear.float()
    quadratic_work = quadratic.float()
    h0 = torch.zeros(
        (batch_size, key_value_heads, value_dimension + 1),
        device=q.device,
        dtype=torch.float32,
    )
    h1 = torch.zeros(
        (batch_size, key_value_heads, head_dimension, value_dimension + 1),
        device=q.device,
        dtype=torch.float32,
    )
    h2 = torch.zeros(
        (batch_size, key_value_heads, pair_count, value_dimension + 1),
        device=q.device,
        dtype=torch.float32,
    )
    d_q = torch.empty_like(q_work, dtype=torch.float32) if need_q else None
    d_constant = torch.zeros_like(constant_work) if need_coefficients else None
    d_linear = torch.zeros_like(linear_work) if need_coefficients else None
    d_quadratic = torch.zeros_like(quadratic_work) if need_coefficients else None
    feature_indices = torch.arange(head_dimension, device=q.device)
    pair_matrix = torch.maximum(feature_indices[:, None], feature_indices[None, :]) * (
        torch.maximum(feature_indices[:, None], feature_indices[None, :]) + 1
    ) // 2 + torch.minimum(feature_indices[:, None], feature_indices[None, :])
    pair_matrix_rows = torch.maximum(feature_indices[:, None], feature_indices[None, :])
    pair_matrix_columns = torch.minimum(
        feature_indices[:, None], feature_indices[None, :]
    )
    pair_row_groups = groups[:, pair_rows]
    pair_column_groups = groups[:, pair_columns]
    linear_by_dimension = linear_work.gather(1, groups)

    for token in range(token_count):
        augmented_value = torch.empty(
            (batch_size, key_value_heads, value_dimension + 1),
            device=q.device,
            dtype=torch.float32,
        )
        augmented_value[..., :value_dimension] = v_work[:, :, token, :]
        augmented_value[..., value_dimension] = 1.0
        key_token = k_work[:, :, token, :]
        h0.add_(augmented_value)
        h1.add_(key_token.unsqueeze(-1) * augmented_value.unsqueeze(-2))
        key_pair = key_token[..., pair_rows] * key_token[..., pair_columns]
        h2.add_(key_pair.unsqueeze(-1) * augmented_value.unsqueeze(-2))

        query_scaled = q_work[:, :, token, :] * float(scale)
        gradient = g_work[:, :, token, :]
        h0_query = h0.index_select(1, query_to_key_value)
        h1_query = h1.index_select(1, query_to_key_value)
        h2_query = h2.index_select(1, query_to_key_value)
        h1_dot = (h1_query * gradient[:, :, None, :]).sum(dim=-1)
        h2_dot = (h2_query * gradient[:, :, None, :]).sum(dim=-1)

        if d_q is not None:
            matrix_row_groups = groups[:, pair_matrix_rows]
            matrix_column_groups = groups[:, pair_matrix_columns]
            quadratic_matrix = quadratic_work[
                torch.arange(query_heads, device=q.device)[:, None, None],
                matrix_row_groups,
                matrix_column_groups,
            ]
            h2_dot_by_feature_pair = h2_dot.index_select(
                2, pair_matrix.reshape(-1)
            ).reshape(batch_size, query_heads, head_dimension, head_dimension)
            quadratic_derivative = (
                2.0
                * quadratic_matrix[None, :, :, :]
                * (query_scaled[:, :, None, :])
                * h2_dot_by_feature_pair
            )
            d_q[:, :, token, :] = float(scale) * (
                linear_by_dimension[None, :, :] * h1_dot
                + quadratic_derivative.sum(dim=-1)
            )

        if d_constant is not None and d_linear is not None and d_quadratic is not None:
            d_constant.add_((h0_query * gradient).sum(dim=(0, 2)))
            linear_contribution = query_scaled * h1_dot
            pair_contribution = (
                multiplicity[None, None, :]
                * query_scaled[..., pair_rows]
                * query_scaled[..., pair_columns]
                * h2_dot
            )
            for group in range(linear_work.shape[-1]):
                feature_mask = groups == group
                d_linear[:, group].add_(
                    (linear_contribution * feature_mask[None, :, :]).sum(dim=(0, 2))
                )
            for row_group in range(linear_work.shape[-1]):
                for column_group in range(linear_work.shape[-1]):
                    pair_mask = (pair_row_groups == row_group) & (
                        pair_column_groups == column_group
                    )
                    d_quadratic[:, row_group, column_group].add_(
                        (pair_contribution * pair_mask[None, :, :]).sum(dim=(0, 2))
                    )
    return d_q, d_constant, d_linear, d_quadratic


def launch_prefix_dq_dcoeff(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    grad_augmented: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    *,
    scale: float,
    need_q: bool,
    need_coefficients: bool,
    coefficient_gradient_mask: tuple[bool, bool, bool] | None = None,
    requested_gradient_mask: tuple[bool, bool, bool, bool, bool, bool] | None = None,
    kernel_plan: KernelPlan | None = None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Run the CUDA prefix schedule, retaining the exact PyTorch fallback."""
    if not need_q and not need_coefficients:
        return None, None, None, None
    _validate_grouped_scan_inputs(
        q,
        k,
        v,
        dim_groups,
        grad_augmented,
        constant,
        linear,
        quadratic,
        scale=scale,
    )
    coefficient_mask = (
        (need_coefficients, need_coefficients, need_coefficients)
        if coefficient_gradient_mask is None
        else tuple(coefficient_gradient_mask)
    )
    if len(coefficient_mask) != 3 or not all(
        isinstance(value, bool) for value in coefficient_mask
    ):
        raise TypeError("coefficient_gradient_mask must contain three A/B/C booleans")
    if any(coefficient_mask) != need_coefficients:
        raise ValueError("need_coefficients must agree with coefficient_gradient_mask")
    full_mask = (
        (need_q, False, False, *coefficient_mask)
        if requested_gradient_mask is None
        else tuple(requested_gradient_mask)
    )
    if len(full_mask) != 6 or not all(isinstance(value, bool) for value in full_mask):
        raise TypeError("requested_gradient_mask must contain six Q/K/V/A/B/C booleans")
    expected_prefix_mask = (need_q, False, False, *coefficient_mask)
    if full_mask != expected_prefix_mask:
        raise ValueError("requested_gradient_mask must agree with prefix outputs")
    if (
        kernel_plan is not None
        and kernel_plan.geometry.requested_gradient_mask != expected_prefix_mask
    ):
        raise ValueError("prefix kernel plan must match the exact prefix gradient mask")
    gmax = linear.shape[-1]
    if q.is_cuda:
        if (
            q.dtype not in {torch.float16, torch.bfloat16}
            or gmax <= 0
            or gmax & (gmax - 1)
            or gmax > 8
            or not triton_prefix_kernels.triton_is_available()
        ):
            raise RuntimeError(
                "CUDA grouped causal prefix requires one planned Triton schedule"
            )
        plan = kernel_plan or build_backward_kernel_plan(
            q,
            k,
            v,
            dim_groups,
            grad_augmented,
            constant,
            linear,
            quadratic,
            needs_input_grad=full_mask,
        )
        return triton_prefix_kernels.grouped_causal_triton_prefix_dq_dcoeff(
            q,
            k,
            v,
            dim_groups,
            grad_augmented,
            constant,
            linear,
            quadratic,
            scale=scale,
            need_q=need_q,
            need_coefficients=need_coefficients,
            coefficient_gradient_mask=coefficient_mask,
            requested_gradient_mask=full_mask,
            plan=plan,
        )
    return _torch_launch_prefix_dq_dcoeff(
        q,
        k,
        v,
        dim_groups,
        grad_augmented,
        constant,
        linear,
        quadratic,
        scale=scale,
        need_q=need_q,
        need_coefficients=need_coefficients,
    )


def _torch_launch_reverse_dk_dv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    grad_augmented: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    *,
    scale: float,
    need_k: bool,
    need_v: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Run the exact KV-owned inclusive reverse scan for dK/dV.

    Query heads mapped to one KV head update A0/A1/A2 in ascending order
    before the current token's state VJP.  That ordering is the causal suffix
    adjoint and makes the native-GQA reduction deterministic without atomics.
    """
    if not need_k and not need_v:
        return None, None
    groups, batch_size, key_value_heads, value_dimension, head_dimension = (
        _validate_grouped_scan_inputs(
            q,
            k,
            v,
            dim_groups,
            grad_augmented,
            constant,
            linear,
            quadratic,
            scale=scale,
        )
    )
    query_heads = q.shape[1]
    token_count = q.shape[2]
    pair_rows, pair_columns = torch.tril_indices(
        head_dimension, head_dimension, device=q.device, dtype=torch.int64
    )
    pair_count = pair_rows.numel()
    multiplicity = torch.where(
        pair_rows == pair_columns,
        torch.ones(pair_count, device=q.device, dtype=torch.float32),
        torch.full((pair_count,), 2.0, device=q.device, dtype=torch.float32),
    )
    q_work = q.float()
    k_work = k.float()
    v_work = v.float()
    g_work = grad_augmented.float()
    constant_work = constant.float()
    linear_work = linear.float()
    quadratic_work = quadratic.float()
    linear_by_dimension = linear_work.gather(1, groups)
    pair_row_groups = groups[:, pair_rows]
    pair_column_groups = groups[:, pair_columns]
    quadratic_by_pair = quadratic_work[
        torch.arange(query_heads, device=q.device)[:, None],
        pair_row_groups,
        pair_column_groups,
    ]
    a0 = (
        torch.zeros(
            (batch_size, key_value_heads, value_dimension + 1),
            device=q.device,
            dtype=torch.float32,
        )
        if need_v
        else None
    )
    a1 = torch.zeros(
        (batch_size, key_value_heads, head_dimension, value_dimension + 1),
        device=q.device,
        dtype=torch.float32,
    )
    a2 = torch.zeros(
        (batch_size, key_value_heads, pair_count, value_dimension + 1),
        device=q.device,
        dtype=torch.float32,
    )
    d_k = torch.empty_like(k_work, dtype=torch.float32) if need_k else None
    d_v = torch.empty_like(v_work, dtype=torch.float32) if need_v else None
    feature_indices = torch.arange(head_dimension, device=q.device)
    pair_matrix = torch.maximum(feature_indices[:, None], feature_indices[None, :]) * (
        torch.maximum(feature_indices[:, None], feature_indices[None, :]) + 1
    ) // 2 + torch.minimum(feature_indices[:, None], feature_indices[None, :])
    group_size = query_heads // key_value_heads

    for token in range(token_count - 1, -1, -1):
        for key_value_head in range(key_value_heads):
            for query_head in range(
                key_value_head * group_size,
                (key_value_head + 1) * group_size,
            ):
                gradient = g_work[:, query_head, token, :]
                query_scaled = q_work[:, query_head, token, :] * float(scale)
                if a0 is not None:
                    a0[:, key_value_head].add_(constant_work[query_head] * gradient)
                a1[:, key_value_head].add_(
                    linear_by_dimension[query_head][None, :, None]
                    * query_scaled[:, :, None]
                    * gradient[:, None, :]
                )
                pair_query = query_scaled[:, pair_rows] * query_scaled[:, pair_columns]
                a2[:, key_value_head].add_(
                    multiplicity[None, :, None]
                    * quadratic_by_pair[query_head][None, :, None]
                    * pair_query[:, :, None]
                    * gradient[:, None, :]
                )

        augmented_value = torch.empty(
            (batch_size, key_value_heads, value_dimension + 1),
            device=q.device,
            dtype=torch.float32,
        )
        augmented_value[..., :value_dimension] = v_work[:, :, token, :]
        augmented_value[..., value_dimension] = 1.0
        key_token = k_work[:, :, token, :]
        if d_v is not None and a0 is not None:
            d_augmented_value = a0 + (key_token.unsqueeze(-1) * a1).sum(dim=2)
            key_pair = key_token[..., pair_rows] * key_token[..., pair_columns]
            d_augmented_value.add_((key_pair.unsqueeze(-1) * a2).sum(dim=2))
            d_v[:, :, token, :] = d_augmented_value[..., :value_dimension]
        if d_k is not None:
            linear_key_gradient = (a1 * augmented_value.unsqueeze(-2)).sum(dim=-1)
            pair_gradient = (a2 * augmented_value.unsqueeze(-2)).sum(dim=-1)
            pair_gradient_matrix = pair_gradient.index_select(
                2, pair_matrix.reshape(-1)
            ).reshape(
                batch_size,
                key_value_heads,
                head_dimension,
                head_dimension,
            )
            quadratic_key_gradient = (
                pair_gradient_matrix * key_token[:, :, None, :]
            ).sum(dim=-1)
            quadratic_key_gradient.add_(
                pair_gradient_matrix.diagonal(dim1=-2, dim2=-1) * key_token
            )
            d_k[:, :, token, :] = linear_key_gradient + quadratic_key_gradient
    return d_k, d_v


def launch_reverse_dk_dv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    grad_augmented: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    *,
    scale: float,
    need_k: bool,
    need_v: bool,
    requested_gradient_mask: tuple[bool, bool, bool, bool, bool, bool] | None = None,
    kernel_plan: KernelPlan | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Run the CUDA suffix schedule, retaining the exact PyTorch fallback."""
    if not need_k and not need_v:
        return None, None
    _validate_grouped_scan_inputs(
        q,
        k,
        v,
        dim_groups,
        grad_augmented,
        constant,
        linear,
        quadratic,
        scale=scale,
    )
    requested_mask = (
        (False, need_k, need_v, False, False, False)
        if requested_gradient_mask is None
        else tuple(requested_gradient_mask)
    )
    expected_suffix_mask = (False, need_k, need_v, False, False, False)
    if requested_mask != expected_suffix_mask:
        raise ValueError("requested_gradient_mask must agree with suffix outputs")
    if (
        kernel_plan is not None
        and kernel_plan.geometry.requested_gradient_mask != expected_suffix_mask
    ):
        raise ValueError("suffix kernel plan must match the exact suffix gradient mask")
    if (
        q.is_cuda
        and q.dtype in {torch.float16, torch.bfloat16}
        and triton_backward_kernels.triton_is_available()
    ):
        plan = kernel_plan or build_backward_kernel_plan(
            q,
            k,
            v,
            dim_groups,
            grad_augmented,
            constant,
            linear,
            quadratic,
            needs_input_grad=requested_mask,
        )
        return triton_backward_kernels.grouped_causal_triton_reverse_dk_dv(
            q,
            k,
            v,
            dim_groups,
            grad_augmented,
            constant,
            linear,
            quadratic,
            scale=scale,
            need_k=need_k,
            need_v=need_v,
            requested_gradient_mask=requested_mask,
            plan=plan,
        )
    return _torch_launch_reverse_dk_dv(
        q,
        k,
        v,
        dim_groups,
        grad_augmented,
        constant,
        linear,
        quadratic,
        scale=scale,
        need_k=need_k,
        need_v=need_v,
    )


def _requires_prefix_vjp(
    needs_input_grad: tuple[bool, bool, bool, bool, bool, bool],
) -> bool:
    """Return whether the inclusive H-prefix owns any requested VJP output.

    The six booleans remain in Q/K/V/A/B/C order.  K/V-only requests must
    never allocate or launch the H-prefix stage; Q or any grouped coefficient
    request requires its exact inclusive recurrence.
    """
    if len(needs_input_grad) != 6 or not all(
        isinstance(value, bool) for value in needs_input_grad
    ):
        raise TypeError("needs_input_grad must be six booleans for Q/K/V/A/B/C")
    return needs_input_grad[0] or any(needs_input_grad[3:])


def _grouped_causal_vjp_from_augmented(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    grad_augmented: torch.Tensor,
    *,
    scale: float,
    needs_input_grad: tuple[bool, bool, bool, bool, bool, bool],
    kernel_plans: GroupedCausalVjpKernelPlans | None = None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Run only the requested causal scans after construction of ``G``."""
    need_q, need_k, need_v, need_constant, need_linear, need_quadratic = (
        needs_input_grad
    )
    need_coefficients = need_constant or need_linear or need_quadratic
    prefix_mask = (need_q, False, False, need_constant, need_linear, need_quadratic)
    suffix_mask = (False, need_k, need_v, False, False, False)
    if (
        kernel_plans is not None
        and kernel_plans.requested_gradient_mask != needs_input_grad
    ):
        raise ValueError("VJP plan bundle does not match the requested gradient mask")
    d_q = d_constant = d_linear = d_quadratic = d_k = d_v = None
    if _requires_prefix_vjp(needs_input_grad):
        prefix_plan = kernel_plans.prefix_plan if kernel_plans is not None else None
        prefix_scope = (
            _observe_grouped_stage("prefix", prefix_plan.to_dict)
            if prefix_plan is not None
            else nullcontext()
        )
        with prefix_scope:
            d_q, d_constant, d_linear, d_quadratic = launch_prefix_dq_dcoeff(
                q,
                k,
                v,
                dim_groups,
                grad_augmented,
                constant,
                linear,
                quadratic,
                scale=scale,
                need_q=need_q,
                need_coefficients=need_coefficients,
                coefficient_gradient_mask=(
                    need_constant,
                    need_linear,
                    need_quadratic,
                ),
                requested_gradient_mask=prefix_mask,
                kernel_plan=prefix_plan,
            )
    if need_k or need_v:
        suffix_plan = kernel_plans.suffix_plan if kernel_plans is not None else None
        suffix_scope = (
            _observe_grouped_stage("suffix", suffix_plan.to_dict)
            if suffix_plan is not None
            else nullcontext()
        )
        with suffix_scope:
            d_k, d_v = launch_reverse_dk_dv(
                q,
                k,
                v,
                dim_groups,
                grad_augmented,
                constant,
                linear,
                quadratic,
                scale=scale,
                need_k=need_k,
                need_v=need_v,
                requested_gradient_mask=suffix_mask,
                kernel_plan=suffix_plan,
            )
    return (
        d_q.to(dtype=q.dtype) if d_q is not None else None,
        d_k.to(dtype=k.dtype) if d_k is not None else None,
        d_v.to(dtype=v.dtype) if d_v is not None else None,
        d_constant if need_constant else None,
        d_linear if need_linear else None,
        d_quadratic if need_quadratic else None,
    )


def grouped_causal_unnormalized_vjp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    grad_numerator: torch.Tensor,
    grad_denominator: torch.Tensor,
    *,
    scale: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Differentiate unnormalized causal outputs with bounded scan state.

    This is intentionally separate from normalization so private diagnostics can
    exercise the same streaming prefix/suffix scans without manufacturing a
    normalized output or calling the legacy materialized-state launcher.
    """
    if not isinstance(grad_numerator, torch.Tensor) or not isinstance(
        grad_denominator, torch.Tensor
    ):
        raise TypeError("unnormalized output gradients must be tensors")
    expected_numerator_shape = (*q.shape[:-1], v.shape[-1])
    expected_denominator_shape = (*q.shape[:-1], 1)
    if grad_numerator.shape != expected_numerator_shape:
        raise ValueError("grad_numerator must have shape [B, Hq, N, DV]")
    if grad_denominator.shape != expected_denominator_shape:
        raise ValueError("grad_denominator must have shape [B, Hq, N, 1]")
    if grad_numerator.device != q.device or grad_denominator.device != q.device:
        raise ValueError("unnormalized output gradients must share the Q device")
    if (
        not grad_numerator.is_floating_point()
        or not grad_denominator.is_floating_point()
    ):
        raise TypeError("unnormalized output gradients must be floating-point")

    gradients = _grouped_causal_vjp_from_augmented(
        q,
        k,
        v,
        dim_groups,
        constant,
        linear,
        quadratic,
        torch.cat(
            (grad_numerator.float(), grad_denominator.float()), dim=-1
        ).contiguous(),
        scale=scale,
        needs_input_grad=(True, True, True, True, True, True),
    )
    if any(gradient is None for gradient in gradients):
        raise AssertionError("the full unnormalized VJP must produce every gradient")
    return gradients  # type: ignore[return-value]


def grouped_causal_first_order_vjp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    grad_output: torch.Tensor | None,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    *,
    grad_numerator: torch.Tensor | None,
    grad_denominator: torch.Tensor | None,
    scale: float,
    needs_input_grad: tuple[bool, bool, bool, bool, bool, bool],
    kernel_plans: GroupedCausalVjpKernelPlans | None = None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Compose one augmented VJP with the requested causal scan sides."""
    if len(needs_input_grad) != 6 or not all(
        isinstance(value, bool) for value in needs_input_grad
    ):
        raise TypeError("needs_input_grad must be six booleans for Q/K/V/A/B/C")
    if not any(needs_input_grad):
        return None, None, None, None, None, None
    normalization_plan = (
        kernel_plans.normalization_plan if kernel_plans is not None else None
    )
    normalization_scope = (
        _observe_grouped_stage("normalization", normalization_plan.to_dict)
        if normalization_plan is not None
        else nullcontext()
    )
    with normalization_scope:
        grad_augmented = augmented_normalization_vjp(
            grad_output,
            numerator,
            denominator,
            grad_numerator=grad_numerator,
            grad_denominator=grad_denominator,
            kernel_plan=normalization_plan,
        )
    _capture_normalization_production_witness(grad_augmented)
    return _grouped_causal_vjp_from_augmented(
        q,
        k,
        v,
        dim_groups,
        constant,
        linear,
        quadratic,
        grad_augmented,
        scale=scale,
        needs_input_grad=needs_input_grad,
        kernel_plans=kernel_plans,
    )


def grouped_causal_backward_diagnostic_witness(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    grad_output: torch.Tensor | None,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    *,
    grad_numerator: torch.Tensor | None,
    grad_denominator: torch.Tensor | None,
    scale: float,
) -> GroupedCausalBackwardDiagnosticWitness:
    """Inspect CPU-only G, H-prefix, and A-suffix stages without a VJP launch.

    The normal backward path neither calls nor saves this record.  It exists
    only to freeze the exact scalar ordering used by the physical recurrence:
    normalized cotangents first, inclusive H before query use, and inclusive A
    before same-token K/V use in descending token order.
    """
    _validate_backward_diagnostic_forward_outputs(q, v, numerator, denominator)
    require_cpu_diagnostic_geometry(q, k, v)
    forward_witness = grouped_causal_forward_diagnostic_witness(
        q, k, v, dim_groups, constant, linear, quadratic, scale=scale
    )
    checked_output = _require_gradient("grad_output", grad_output, reference=numerator)
    checked_numerator = _require_gradient(
        "grad_numerator", grad_numerator, reference=numerator
    )
    checked_denominator = _require_gradient(
        "grad_denominator", grad_denominator, reference=denominator
    )
    with torch.no_grad():
        g_work = _torch_augmented_normalization_vjp(
            checked_output,
            numerator.detach(),
            denominator.detach(),
            checked_numerator,
            checked_denominator,
        )
        query_heads = q.shape[1]
        key_value_heads = k.shape[1]
        token_count = q.shape[2]
        head_dimension = q.shape[-1]
        value_dimension = v.shape[-1]
        groups = canonical_group_indices(
            dim_groups,
            query_heads=query_heads,
            head_dimension=head_dimension,
            gmax=linear.shape[-1],
        )
        pair_rows, pair_columns, multiplicity = canonical_packed_pair_metadata(
            head_dimension, device=q.device
        )
        pair_count = pair_rows.numel()
        q_work = q.detach().float()
        constant_work = constant.detach().float()
        linear_work = linear.detach().float()
        quadratic_work = quadratic.detach().float()
        linear_by_dimension = linear_work.gather(1, groups)
        pair_row_groups = groups[:, pair_rows]
        pair_column_groups = groups[:, pair_columns]
        quadratic_by_pair = quadratic_work[
            torch.arange(query_heads, device=q.device)[:, None],
            pair_row_groups,
            pair_column_groups,
        ]
        batch_size = q.shape[0]
        a0 = torch.zeros(
            (batch_size, key_value_heads, value_dimension + 1),
            device=q.device,
            dtype=torch.float32,
        )
        a1 = torch.zeros(
            (batch_size, key_value_heads, head_dimension, value_dimension + 1),
            device=q.device,
            dtype=torch.float32,
        )
        a2 = torch.zeros(
            (batch_size, key_value_heads, pair_count, value_dimension + 1),
            device=q.device,
            dtype=torch.float32,
        )
        group_size = query_heads // key_value_heads
        suffix_a: list[GroupedCausalSuffixStateSnapshot] = []
        for token in range(token_count - 1, -1, -1):
            for key_value_head in range(key_value_heads):
                for query_head in range(
                    key_value_head * group_size,
                    (key_value_head + 1) * group_size,
                ):
                    gradient = g_work[:, query_head, token, :]
                    query_scaled = q_work[:, query_head, token, :] * float(scale)
                    a0[:, key_value_head].add_(constant_work[query_head] * gradient)
                    a1[:, key_value_head].add_(
                        linear_by_dimension[query_head][None, :, None]
                        * query_scaled[:, :, None]
                        * gradient[:, None, :]
                    )
                    pair_query = (
                        query_scaled[:, pair_rows] * query_scaled[:, pair_columns]
                    )
                    a2[:, key_value_head].add_(
                        multiplicity[None, :, None]
                        * quadratic_by_pair[query_head][None, :, None]
                        * pair_query[:, :, None]
                        * gradient[:, None, :]
                    )
            suffix_a.append(
                GroupedCausalSuffixStateSnapshot(
                    token_index=token,
                    a0=a0.clone(),
                    a1=a1.clone(),
                    a2=a2.clone(),
                )
            )
        prefix_h = tuple(
            GroupedCausalStateSnapshot(
                token_index=stage.token_index,
                h0=stage.h0_inclusive.clone(),
                h1=stage.h1_inclusive.clone(),
                h2=stage.h2_inclusive.clone(),
            )
            for stage in forward_witness.tokens
        )
        return GroupedCausalBackwardDiagnosticWitness(
            physical_path=PHYSICAL_PATH_IDENTIFIER,
            pair_rows=pair_rows.clone(),
            pair_columns=pair_columns.clone(),
            pair_multiplicity=multiplicity.clone(),
            normalization_g=g_work.clone(),
            prefix_h=prefix_h,
            suffix_a=tuple(suffix_a),
        )


__all__ = (
    "GroupedCausalVjpKernelPlans",
    "augmented_normalization_vjp",
    "build_backward_kernel_plan",
    "build_backward_kernel_plan_from_metadata",
    "build_causal_vjp_kernel_plans_from_metadata",
    "build_normalization_kernel_plan_for_geometry",
    "build_normalization_kernel_plan_from_metadata",
    "grouped_causal_backward_diagnostic_witness",
    "grouped_causal_first_order_vjp",
    "grouped_causal_unnormalized_vjp",
    "normalization_production_witness_capture_names",
    "launch_prefix_dq_dcoeff",
    "launch_reverse_dk_dv",
    "triton_is_available",
)
