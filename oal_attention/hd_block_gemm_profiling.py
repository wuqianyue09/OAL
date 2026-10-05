"""Opt-in stage labels and lightweight execution counters for HD profiling."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
import inspect
from typing import Iterator

import torch

_HD_STAGE_PROFILING_ENABLED: ContextVar[bool] = ContextVar(
    "hd_stage_profiling_enabled",
    default=False,
)
_HD_PROFILE_STATE: ContextVar["_HDProfileState | None"] = ContextVar(
    "hd_profile_state",
    default=None,
)
_HD_STAGE_STACK: ContextVar[tuple[str, ...]] = ContextVar(
    "hd_stage_stack",
    default=(),
)
_DISABLED_STAGE = nullcontext()
_ALLOCATION_CATEGORIES = frozenset(
    (
        "feature",
        "fold",
        "carry",
        "gqa_expansion",
        "saved_activation",
        "other",
    )
)
_QUERY_DS_STAGE = "hd.backward.query_ds"
_QUERY_DPHI_GLOBAL_STAGE = "hd.backward.query_dphi_global"
_QUERY_DPHI_LOCAL_STAGE = "hd.backward.query_dphi_local"
_QUERY_DPHI_MERGE_STAGE = "hd.backward.query_dphi_merge"
_QUERY_CONTRACTION_PRODUCER_STAGES = (
    _QUERY_DS_STAGE,
    _QUERY_DPHI_GLOBAL_STAGE,
    _QUERY_DPHI_LOCAL_STAGE,
    _QUERY_DPHI_MERGE_STAGE,
)
_QUERY_CONTRACTIONS_LEGACY_STAGE = "hd.backward.query_contractions"
_SHARED_K_FEATURE_STAGE = "hd.backward.shared_k_feature"
_SHARED_DS_STAGE = "hd.backward.shared_ds"
_SHARED_GQA_COPY_STAGE = "hd.backward.shared_gqa_copy"

# Every planned buffer belongs to one named evidence bucket.  These are exact
# logical-buffer identities rather than heuristics over a string name: a
# future buffer must deliberately choose a category at its allocation seam.
_FEATURE_ALLOCATION_BUFFERS = frozenset(
    (
        "forward_phi_q_input_work",
        "forward_phi_k_input_work",
        "forward_pair_work",
        "forward_pair_second_work",
        "forward_phi_k_cache",
        "forward_phi_k_wave",
        "forward_phi_q_wave",
        "forward_u_wave",
        "backward_feature_pair_work",
        "backward_feature_pair_second_work",
        "backward_feature_k_input_work",
        "backward_phi_k_wave",
        "backward_phi_q_wave",
        "backward_u_wave",
        "backward_query_input_work",
    )
)
_FOLD_ALLOCATION_BUFFERS = frozenset(
    (
        "backward_dphi_q",
        "backward_dphi_q_local",
        "backward_dphi_k",
        "backward_dphi_k_cross",
        "backward_query_fold_source",
        "backward_query_fold_second",
        "backward_query_fold_linear_source",
        "backward_query_fold_output",
        "backward_key_input_work",
        "backward_key_fold_source",
        "backward_key_fold_output",
        "grad_q",
        "grad_k",
    )
)
_CARRY_ALLOCATION_BUFFERS = frozenset(
    (
        "block_totals_carry",
        "forward_block_totals_wave",
        "forward_scan_inclusive",
        "backward_dcarry",
        "backward_right_scan_dt",
        "backward_right_scan_total",
        "backward_dt_wave_tc",
    )
)
_GQA_EXPANSION_ALLOCATION_BUFFERS = frozenset(
    (
        "forward_phi_k_query_wave",
        "forward_carry_query_wave",
        "forward_u_query_wave",
        "backward_phi_k_query_wave",
        "backward_carry_query_wave",
        "backward_u_query_wave",
        "backward_dphi_k_query_wave",
        "backward_du_query_wave",
        "backward_dcarry_query_wave",
    )
)
_SAVED_ACTIVATION_ALLOCATION_BUFFERS = frozenset(
    ("saved_local_score_tc", "saved_phi_k_tc")
)


@dataclass
class _HDProfileState:
    capture_cuda_intervals: bool = True
    stage_calls: dict[str, int] = field(default_factory=dict)
    allocations: dict[str, list[int]] = field(default_factory=dict)
    contractions: dict[
        tuple[
            str,
            tuple[int, int, int, int],
            tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
            tuple[str, str, str],
        ],
        int,
    ] = field(default_factory=dict)
    cuda_stage_intervals: dict[str, list[tuple[object, object]]] = field(
        default_factory=dict
    )


def _empty_snapshot() -> dict[str, object]:
    return {
        "schema_version": "hd_block_gemm_profile_counters_v3",
        "allocation_api_calls": 0,
        "allocation_bytes": 0,
        "allocations": [],
        "contractions": [],
        "stage_calls": {},
        "cuda_stage_times_us": {},
    }


@contextmanager
def _enable_hd_stage_profiling(
    *,
    capture_cuda_intervals: bool = True,
) -> Iterator[None]:
    """Enable profiling-only labels and counters inside the current context."""
    enabled_token = _HD_STAGE_PROFILING_ENABLED.set(True)
    state_token = _HD_PROFILE_STATE.set(
        _HDProfileState(capture_cuda_intervals=capture_cuda_intervals)
    )
    stack_token = _HD_STAGE_STACK.set(())
    try:
        yield
    finally:
        _HD_STAGE_STACK.reset(stack_token)
        _HD_PROFILE_STATE.reset(state_token)
        _HD_STAGE_PROFILING_ENABLED.reset(enabled_token)


def _capture_hd_stage_profiling_state() -> _HDProfileState | None:
    """Capture the active profile state for a deferred autograd callback."""
    if not _HD_STAGE_PROFILING_ENABLED.get():
        return None
    return _HD_PROFILE_STATE.get()


@contextmanager
def _resume_hd_stage_profiling_state(
    state: _HDProfileState | None,
) -> Iterator[None]:
    """Rebind a captured state when autograd enters a fresh Python context."""
    if state is None:
        yield
        return
    enabled_token = _HD_STAGE_PROFILING_ENABLED.set(True)
    state_token = _HD_PROFILE_STATE.set(state)
    stack_token = _HD_STAGE_STACK.set(())
    try:
        yield
    finally:
        _HD_STAGE_STACK.reset(stack_token)
        _HD_PROFILE_STATE.reset(state_token)
        _HD_STAGE_PROFILING_ENABLED.reset(enabled_token)


@contextmanager
def _profiled_stage(name: str) -> Iterator[None]:
    stack_token = _HD_STAGE_STACK.set(_HD_STAGE_STACK.get() + (name,))
    state = _HD_PROFILE_STATE.get()
    if state is not None:
        state.stage_calls[name] = state.stage_calls.get(name, 0) + 1
    start_event: object | None = None
    end_event: object | None = None
    if state is not None and state.capture_cuda_intervals and torch.cuda.is_available():
        # key_averages() can report zero CUDA time for record_function ranges
        # even though the raw trace contains the enclosed kernels.  A matching
        # pair of CUDA events is the authoritative device interval here.
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    try:
        with torch.autograd.profiler.record_function(name):
            yield
    finally:
        if start_event is not None and end_event is not None and state is not None:
            end_event.record()
            state.cuda_stage_intervals.setdefault(name, []).append(
                (start_event, end_event)
            )
        _HD_STAGE_STACK.reset(stack_token)


def _record_hd_stage(name: str) -> object:
    """Return a reusable no-op unless profiling was explicitly enabled."""
    if not _HD_STAGE_PROFILING_ENABLED.get():
        return _DISABLED_STAGE
    return _profiled_stage(name)


def _planned_allocation_category(name: str) -> str:
    """Return the explicit evidence category for one logical plan buffer."""
    if not isinstance(name, str) or not name:
        raise ValueError("planned allocation name must be a non-empty string")
    if name in _FEATURE_ALLOCATION_BUFFERS:
        return "feature"
    if name in _FOLD_ALLOCATION_BUFFERS:
        return "fold"
    if name in _CARRY_ALLOCATION_BUFFERS:
        return "carry"
    if name in _GQA_EXPANSION_ALLOCATION_BUFFERS:
        return "gqa_expansion"
    if name in _SAVED_ACTIVATION_ALLOCATION_BUFFERS:
        return "saved_activation"
    return "other"


def _record_hd_allocation(
    name: str,
    tensor: torch.Tensor,
    *,
    category: str,
) -> torch.Tensor:
    """Account for a planned allocation only inside an explicit profile scope."""
    if category not in _ALLOCATION_CATEGORIES:
        raise ValueError("unsupported HD allocation category")
    state = _HD_PROFILE_STATE.get()
    if state is None:
        return tensor
    record = state.allocations.setdefault(f"{category}\0{name}", [0, 0])
    record[0] += 1
    record[1] += tensor.numel() * tensor.element_size()
    return tensor


def _current_hd_contraction_role() -> str:
    """Return the current logical BMM owner, including unwrapped call sites."""
    stack = _HD_STAGE_STACK.get()
    if stack:
        return stack[-1]
    frame = inspect.currentframe()
    helper_caller = frame.f_back if frame is not None else None
    caller = helper_caller.f_back if helper_caller is not None else None
    module = str(caller.f_globals.get("__name__", "unknown")) if caller else "unknown"
    function = caller.f_code.co_name if caller else "unknown"
    return f"hd.contraction.{module.rsplit('.', maxsplit=1)[-1]}.{function}"


def _record_hd_contraction(
    left: torch.Tensor,
    right: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Record BMM geometry against the caller's current named HD stage."""
    state = _HD_PROFILE_STATE.get()
    if state is None:
        return
    if left.ndim != 3 or right.ndim != 3 or out.ndim != 3:
        # Instrumentation must not change the error contract of the real
        # contraction facade; unusual inputs are simply reported generically.
        return
    role = _current_hd_contraction_role()
    shape = (
        int(left.shape[0]),
        int(left.shape[1]),
        int(right.shape[2]),
        int(left.shape[2]),
    )
    data_types = (
        str(left.dtype).removeprefix("torch."),
        str(right.dtype).removeprefix("torch."),
        str(out.dtype).removeprefix("torch."),
    )
    strides = (
        tuple(int(value) for value in left.stride()),
        tuple(int(value) for value in right.stride()),
        tuple(int(value) for value in out.stride()),
    )
    key = (role, shape, strides, data_types)
    state.contractions[key] = state.contractions.get(key, 0) + 1


def _record_hd_bmm_stage(
    left: torch.Tensor,
    right: torch.Tensor,
    out: torch.Tensor,
) -> object:
    """Create a BMM-only range for throughput evidence in profile mode."""
    if not _HD_STAGE_PROFILING_ENABLED.get():
        return _DISABLED_STAGE
    role = _current_hd_contraction_role()
    return _profiled_stage(
        "hd.contraction.bmm."
        f"{role}.b{left.shape[0]}.m{left.shape[1]}.n{right.shape[2]}.k{left.shape[2]}"
    )


def _profile_snapshot() -> dict[str, object]:
    """Return canonical profile counters for the active context, if any."""
    state = _HD_PROFILE_STATE.get()
    if state is None:
        return _empty_snapshot()
    allocations = []
    for key, counts in sorted(state.allocations.items()):
        category, name = key.split("\0", maxsplit=1)
        allocations.append(
            {
                "logical_buffer": name,
                "category": category,
                "api_calls": counts[0],
                "bytes": counts[1],
            }
        )
    contractions = [
        {
            "role": role,
            "shape": list(shape),
            "strides": [list(stride) for stride in strides],
            "data_types": list(data_types),
            "calls": calls,
        }
        for (role, shape, strides, data_types), calls in sorted(
            state.contractions.items()
        )
    ]
    cuda_stage_times_us: dict[str, float] = {}
    for name, intervals in state.cuda_stage_intervals.items():
        total_us = 0.0
        for start_event, end_event in intervals:
            # The profile runner synchronizes before taking its snapshot.  Keep
            # this defensive so instrumentation never changes the operator's
            # error contract if a caller snapshots an incomplete stream.
            try:
                total_us += float(start_event.elapsed_time(end_event)) * 1_000.0
            except RuntimeError:
                continue
        cuda_stage_times_us[name] = total_us
    return {
        "schema_version": "hd_block_gemm_profile_counters_v3",
        "allocation_api_calls": sum(int(record["api_calls"]) for record in allocations),
        "allocation_bytes": sum(int(record["bytes"]) for record in allocations),
        "allocations": allocations,
        "contractions": contractions,
        "stage_calls": {
            name: state.stage_calls[name] for name in sorted(state.stage_calls)
        },
        "cuda_stage_times_us": {
            name: cuda_stage_times_us[name] for name in sorted(cuda_stage_times_us)
        },
    }


__all__: tuple[str, ...] = ()
