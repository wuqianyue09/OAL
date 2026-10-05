"""Shared contracts for KV-owned grouped causal streaming scans.

This module intentionally contains only CPU-testable geometry and workspace
accounting.  Triton launchers live in the causal forward/backward modules so
the scan layout can be audited without importing CUDA tooling.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json
from math import floor, prod
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

import torch

_FP32_BYTES = 4
_SUPPORTED_TOKEN_BLOCKS = frozenset((8, 16, 32))
_SUPPORTED_PAIR_BLOCKS = frozenset((32, 64, 128))
_SUPPORTED_VALUE_BLOCKS = frozenset((8, 16, 32))
_SUPPORTED_GROUPS_PER_WAVE = frozenset((1, 2, 4, 8))
_SUPPORTED_REDUCTION_REPLICAS = frozenset((1, 2, 4, 8))
_MAX_AUXILIARY_WORKSPACE_BYTES = 512 * 2**20
_DEFAULT_PLANNER_VERSION = "grouped_causal_workspace_v3"
# ``augmented_normalization_vjp`` launches fixed 16-value tiles.  This is
# shared with its Triton module so the dedicated evidence workspace cannot
# silently be planned with a different denominator-partial shape.
NORMALIZATION_VALUE_BLOCK = 16

# Stable CUDA Driver ``CUdevice_attribute`` enum values.  These identify what
# to query; they do not encode a GPU model, memory capacity, or architecture.
_CUDA_DRIVER_ATTRIBUTE_CODES = MappingProxyType(
    {
        "sm_count": 16,
        "warp_size": 10,
        "max_threads_per_block": 1,
        "max_threads_per_sm": 39,
        "shared_memory_per_block_optin": 97,
        "shared_memory_per_sm": 81,
        "registers_per_block": 12,
        "registers_per_sm": 82,
        "l2_bytes": 38,
        "memory_bus_width_bits": 37,
        "memory_clock_rate": 36,
    }
)
_CUDA_RUNTIME_PROPERTY_ALIASES = MappingProxyType(
    {
        "sm_count": ("multiprocessor_count", "multi_processor_count"),
        "warp_size": ("warpSize", "warp_size"),
        "max_threads_per_block": ("max_threads_per_block", "maxThreadsPerBlock"),
        "max_threads_per_sm": (
            "max_threads_per_multi_processor",
            "max_threads_per_sm",
            "maxThreadsPerMultiProcessor",
        ),
        "shared_memory_per_block_optin": (
            "max_shared_mem",
            "max_shared_memory_per_block_optin",
            "shared_memory_per_block_optin",
            "shared_memory_per_block",
        ),
        "shared_memory_per_sm": (
            "max_shared_memory_per_multiprocessor",
            "shared_memory_per_multiprocessor",
            "shared_memory_per_sm",
        ),
        "registers_per_block": (
            "max_num_regs",
            "max_registers_per_block",
            "regs_per_block",
            "registers_per_block",
        ),
        "registers_per_sm": (
            "max_registers_per_multiprocessor",
            "regs_per_multiprocessor",
            "registers_per_sm",
        ),
        "l2_bytes": ("L2_cache_size", "l2_cache_size", "l2_bytes"),
        "memory_bus_width_bits": ("mem_bus_width", "memory_bus_width"),
        "memory_clock_rate": ("mem_clock_rate", "memory_clock_rate"),
    }
)

_N_INDEPENDENT = "n_independent"
_N_DEPENDENT_ALLOWLIST = "n_dependent_allowlist"
_N_DEPENDENT_ALLOWLIST_NAMES = frozenset(
    (
        "q",
        "k",
        "v",
        "output",
        "numerator",
        "denominator",
        "d_output",
        "d_numerator",
        "d_denominator",
        "dq",
        "dk",
        "dv",
        "normalization_g",
        "normalization_numerator",
        "normalization_denominator",
        "normalization_grad_output",
        "normalization_grad_numerator",
        "normalization_grad_denominator",
        "normalization_denominator_partials",
        "denominator_partial",
        "saved_q",
        "saved_k",
        "saved_v",
        "saved_output",
        "saved_numerator",
        "saved_denominator",
    )
)
_EXTERNAL_ALIAS_TARGETS = frozenset(
    (
        "dim_groups",
        "constant",
        "linear",
        "quadratic",
        "numerator",
        "denominator",
        "grad_output",
        "grad_numerator",
        "grad_denominator",
    )
)
_PAIR_METADATA_CONTRACTS = (
    ("prefix_pair_metadata", "prefix_pair_rows", "prefix_pair_columns"),
    ("suffix_pair_metadata", "suffix_pair_rows", "suffix_pair_columns"),
)
_LIFETIME_NAMES = {
    "forward": (0, 2),
    "normalization": (3, 4),
    "prefix": (5, 7),
    "suffix": (8, 10),
    "saved": (0, 10),
}
_DTYPE_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
    "int32": 4,
    "int64": 8,
    "bool": 1,
}

# This marker names the mathematical state transition only.  It is deliberately
# not a reviewed capability or a production-admission record.
PHYSICAL_PATH_IDENTIFIER = "hadamard_h012_packed_diag_v1"
DIAGNOSTIC_SEQUENCE_CAP = 8
DIAGNOSTIC_MACRO_TOKEN_BLOCK = 2
DIAGNOSTIC_RECORD_BYTE_BUDGET = 8 * 1024 * 1024


def _canonical_dtype_name(dtype: torch.dtype | str) -> str:
    """Return a JSON-safe dtype spelling used by planning records."""
    if isinstance(dtype, torch.dtype):
        name = str(dtype).removeprefix("torch.")
    elif isinstance(dtype, str):
        name = dtype.removeprefix("torch.")
    else:
        raise TypeError("dtype must be a torch.dtype or canonical dtype string")
    if name not in _DTYPE_BYTES:
        raise TypeError(f"unsupported workspace dtype: {name}")
    return name


def _stable_json(payload: Mapping[str, Any]) -> str:
    """Serialize plan data without runtime-dependent formatting."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _positive_property_value(source: object, names: tuple[str, ...]) -> int | None:
    """Return the first positive integer under an object attribute or map key."""
    for name in names:
        if isinstance(source, Mapping):
            value = source.get(name)
        else:
            value = getattr(source, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _triton_runtime_device_properties(
    triton_module: object | None,
    *,
    device_index: int,
) -> Mapping[str, object]:
    """Read Triton's per-device Driver dictionary without making it mandatory.

    Triton 3.2 exposes this through
    ``triton.runtime.driver.active.utils.get_device_properties``.  The exact
    dictionary is intentionally treated as versioned runtime data, so missing
    keys fall through to PyTorch aliases and then the CUDA Driver API.
    """
    if triton_module is None:
        return {}
    try:
        runtime = getattr(triton_module, "runtime")
        driver = getattr(runtime, "driver")
        active_driver = getattr(driver, "active")
        utils = getattr(active_driver, "utils")
        getter = getattr(utils, "get_device_properties")
        runtime_properties = getter(device_index)
    except (AttributeError, RuntimeError, TypeError, ValueError, OSError):
        return {}
    return runtime_properties if isinstance(runtime_properties, Mapping) else {}


def _cuda_driver_attribute(device_index: int, field_name: str) -> int | None:
    """Dynamically query one CUDA Driver attribute when higher layers omit it."""
    attribute_code = _CUDA_DRIVER_ATTRIBUTE_CODES[field_name]
    library = None
    for library_name in ("libcuda.so.1", "libcuda.so", "nvcuda.dll"):
        try:
            library = ctypes.CDLL(library_name)
            break
        except OSError:
            continue
    if library is None:
        return None
    try:
        cu_init = library.cuInit
        cu_device_get = library.cuDeviceGet
        cu_device_get_attribute = library.cuDeviceGetAttribute
    except AttributeError:
        return None
    cu_init.argtypes = [ctypes.c_uint]
    cu_init.restype = ctypes.c_int
    cu_device_get.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cu_device_get.restype = ctypes.c_int
    cu_device_get_attribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    cu_device_get_attribute.restype = ctypes.c_int
    if cu_init(0) != 0:
        return None
    device = ctypes.c_int()
    if cu_device_get(ctypes.byref(device), device_index) != 0:
        return None
    value = ctypes.c_int()
    if cu_device_get_attribute(ctypes.byref(value), attribute_code, device.value) != 0:
        return None
    return value.value if value.value > 0 else None


def collect_cuda_device_properties(
    torch_properties: object,
    *,
    device_index: int,
    triton_module: object | None = None,
    driver_attribute_reader: Callable[[int, str], int | None] | None = None,
) -> dict[str, int]:
    """Collect one complete, dynamic CUDA planner hardware record.

    The order is deliberate and shared by forward, prefix VJP, suffix VJP,
    and normalization VJP: Triton's live CUDA Driver dictionary first,
    PyTorch's exposed aliases second, then a direct CUDA Driver query.  This
    prevents sparse PyTorch property wrappers from silently changing a plan's
    identity while still failing closed if a required live attribute cannot be
    discovered.
    """
    if (
        not isinstance(device_index, int)
        or isinstance(device_index, bool)
        or device_index < 0
    ):
        raise ValueError("CUDA device index must be a non-negative integer")
    runtime_properties = _triton_runtime_device_properties(
        triton_module,
        device_index=device_index,
    )
    read_driver_attribute = driver_attribute_reader or _cuda_driver_attribute
    resolved: dict[str, int] = {}
    for field_name, aliases in _CUDA_RUNTIME_PROPERTY_ALIASES.items():
        value = _positive_property_value(runtime_properties, aliases)
        if value is None:
            value = _positive_property_value(torch_properties, aliases)
        if value is None:
            value = read_driver_attribute(device_index, field_name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RuntimeError(
                "CUDA device properties omit a positive "
                f"{field_name} after Triton, PyTorch, and CUDA Driver queries"
            )
        resolved[field_name] = value
    return resolved


def cuda_device_index(device: torch.device) -> int:
    """Resolve ``cuda`` to its live ordinal without recording device identity."""
    index = device.index
    return torch.cuda.current_device() if index is None else index


@lru_cache(maxsize=1)
def _planner_source_hash() -> str:
    """Hash the planner source itself; no device/product data participates."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _normalized_lifetime(lifetime: tuple[int, int] | str) -> tuple[int, int]:
    if isinstance(lifetime, str):
        try:
            return _LIFETIME_NAMES[lifetime]
        except KeyError as error:
            raise ValueError(f"unknown workspace lifetime: {lifetime}") from error
    if (
        not isinstance(lifetime, tuple)
        or len(lifetime) != 2
        or any(not isinstance(point, int) for point in lifetime)
        or lifetime[0] < 0
        or lifetime[0] > lifetime[1]
    ):
        raise ValueError("lifetime must be a non-negative inclusive (start, end) pair")
    return lifetime


@dataclass(frozen=True)
class KernelGeometry:
    """Every currently-known planner/dispatcher discriminator.

    This deliberately has hardware resource and toolchain fields, but never a
    product name, device UUID, PCI identity, or model identity.  Available
    memory is represented only by a dispatch-time budget and is never queried
    from this CPU-testable planner.
    """

    compute_capability: tuple[int, int]
    sm_count: int
    warp_size: int
    max_threads_per_block: int
    max_threads_per_sm: int
    shared_memory_per_block_optin: int
    shared_memory_per_sm: int
    registers_per_block: int
    registers_per_sm: int
    l2_bytes: int
    memory_bus_width_bits: int
    memory_clock_rate: int
    torch_version: str
    triton_version: str
    source_hash: str
    dtype: torch.dtype | str
    batch_size: int
    query_heads: int
    key_value_heads: int
    sequence_length: int
    head_dimension: int
    value_dimension: int
    gmax: int
    causal: bool
    group_layout: str
    coefficient_layout: str
    stride_class: str
    requested_gradient_mask: tuple[bool, ...]
    workspace_budget_bytes: int | None = None
    # The default composite path records the normal forward/VJP plan.  The
    # normalization VJP gets its own plan because it launches different
    # kernels and owns a different FP32 scratch contract; it never changes
    # the KV-owned H0/H1/H2 Hadamard-diagonal recurrence.
    launch_stage: str = "composite"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.compute_capability, tuple)
            or len(self.compute_capability) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in self.compute_capability
            )
            or any(value < 0 for value in self.compute_capability)
        ):
            raise ValueError("compute_capability must be a two-item non-negative tuple")
        _require_positive(
            sm_count=self.sm_count,
            warp_size=self.warp_size,
            max_threads_per_block=self.max_threads_per_block,
            max_threads_per_sm=self.max_threads_per_sm,
            shared_memory_per_block_optin=self.shared_memory_per_block_optin,
            shared_memory_per_sm=self.shared_memory_per_sm,
            registers_per_block=self.registers_per_block,
            registers_per_sm=self.registers_per_sm,
            l2_bytes=self.l2_bytes,
            memory_bus_width_bits=self.memory_bus_width_bits,
            memory_clock_rate=self.memory_clock_rate,
            batch_size=self.batch_size,
            query_heads=self.query_heads,
            key_value_heads=self.key_value_heads,
            sequence_length=self.sequence_length,
            head_dimension=self.head_dimension,
            value_dimension=self.value_dimension,
            gmax=self.gmax,
        )
        if self.query_heads % self.key_value_heads:
            raise ValueError("query_heads must be divisible by key_value_heads")
        if not isinstance(self.causal, bool):
            raise TypeError("causal must be a bool")
        for name in (
            "torch_version",
            "triton_version",
            "source_hash",
            "group_layout",
            "coefficient_layout",
            "stride_class",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} is a required dispatch discriminator")
        if self.group_layout not in {"shared", "per_head"}:
            raise ValueError("group_layout must be shared or per_head")
        if self.coefficient_layout not in {"shared", "per_head"}:
            raise ValueError("coefficient_layout must be shared or per_head")
        normalized_mask = tuple(self.requested_gradient_mask)
        if len(normalized_mask) == 4:
            # The public factor input expands into A/B/C before this planner.
            normalized_mask = (
                normalized_mask[0],
                normalized_mask[1],
                normalized_mask[2],
                normalized_mask[3],
                normalized_mask[3],
                normalized_mask[3],
            )
        if len(normalized_mask) != 6 or any(
            not isinstance(value, bool) for value in normalized_mask
        ):
            raise TypeError(
                "requested_gradient_mask must contain six booleans for Q/K/V/A/B/C"
            )
        if self.workspace_budget_bytes is not None and (
            not isinstance(self.workspace_budget_bytes, int)
            or isinstance(self.workspace_budget_bytes, bool)
            or self.workspace_budget_bytes <= 0
        ):
            raise ValueError("workspace_budget_bytes must be a positive integer")
        if self.launch_stage not in {"composite", "normalization_vjp"}:
            raise ValueError("launch_stage must be composite or normalization_vjp")
        object.__setattr__(self, "dtype", _canonical_dtype_name(self.dtype))
        object.__setattr__(self, "requested_gradient_mask", normalized_mask)

    @property
    def B(self) -> int:
        return self.batch_size

    @property
    def Hq(self) -> int:
        return self.query_heads

    @property
    def Hkv(self) -> int:
        return self.key_value_heads

    @property
    def N(self) -> int:
        return self.sequence_length

    @property
    def D(self) -> int:
        return self.head_dimension

    @property
    def DV(self) -> int:
        return self.value_dimension

    @property
    def pair_count(self) -> int:
        return self.head_dimension * (self.head_dimension + 1) // 2

    @property
    def augmented_value_dimension(self) -> int:
        return self.value_dimension + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "compute_capability": list(self.compute_capability),
            "sm_count": self.sm_count,
            "warp_size": self.warp_size,
            "max_threads_per_block": self.max_threads_per_block,
            "max_threads_per_sm": self.max_threads_per_sm,
            "shared_memory_per_block_optin": self.shared_memory_per_block_optin,
            "shared_memory_per_sm": self.shared_memory_per_sm,
            "registers_per_block": self.registers_per_block,
            "registers_per_sm": self.registers_per_sm,
            "l2_bytes": self.l2_bytes,
            "memory_bus_width_bits": self.memory_bus_width_bits,
            "memory_clock_rate": self.memory_clock_rate,
            "torch_version": self.torch_version,
            "triton_version": self.triton_version,
            "source_hash": self.source_hash,
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "query_heads": self.query_heads,
            "key_value_heads": self.key_value_heads,
            "sequence_length": self.sequence_length,
            "head_dimension": self.head_dimension,
            "value_dimension": self.value_dimension,
            "gmax": self.gmax,
            "causal": self.causal,
            "group_layout": self.group_layout,
            "coefficient_layout": self.coefficient_layout,
            "stride_class": self.stride_class,
            "requested_gradient_mask": list(self.requested_gradient_mask),
            "workspace_budget_bytes": self.workspace_budget_bytes,
            "launch_stage": self.launch_stage,
        }

    def to_json(self) -> str:
        return _stable_json(self.to_dict())


@dataclass(frozen=True)
class WorkspaceAllocation:
    """One concrete workspace or alias record with a checked byte formula."""

    name: str
    dtype: torch.dtype | str
    shape: tuple[int, ...]
    axes: tuple[str, ...]
    lifetime: tuple[int, int] | str
    n_dependency: str
    alias_of: str | None = None
    formula: str = ""
    bytes: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("workspace allocation name must be non-empty")
        normalized_shape = tuple(self.shape)
        normalized_axes = tuple(self.axes)
        if len(normalized_shape) != len(normalized_axes):
            raise ValueError(
                "workspace allocation shape and axes must have equal length"
            )
        if not normalized_shape or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in normalized_shape
        ):
            raise ValueError(
                "workspace allocation shape must contain positive integers"
            )
        if any(not isinstance(axis, str) or not axis for axis in normalized_axes):
            raise ValueError("workspace allocation axes must be non-empty strings")
        if self.n_dependency not in {_N_INDEPENDENT, _N_DEPENDENT_ALLOWLIST}:
            raise ValueError("n_dependency must classify an allocation explicitly")
        if self.alias_of is not None and (
            not isinstance(self.alias_of, str) or not self.alias_of
        ):
            raise ValueError("alias_of must be a non-empty allocation/input name")
        dtype_name = _canonical_dtype_name(self.dtype)
        calculated_bytes = prod(normalized_shape) * _DTYPE_BYTES[dtype_name]
        if self.bytes is not None and self.bytes != calculated_bytes:
            raise ValueError(
                "workspace allocation byte formula differs from dtype and concrete shape"
            )
        if not isinstance(self.formula, str):
            raise TypeError("workspace allocation formula must be a string")
        formula = self.formula or (
            f"{_DTYPE_BYTES[dtype_name]} * " + " * ".join(normalized_axes)
        )
        if self.alias_of is not None:
            formula = f"alias_of({self.alias_of}); {formula}"
        object.__setattr__(self, "dtype", dtype_name)
        object.__setattr__(self, "shape", normalized_shape)
        object.__setattr__(self, "axes", normalized_axes)
        object.__setattr__(self, "lifetime", _normalized_lifetime(self.lifetime))
        object.__setattr__(self, "formula", formula)
        object.__setattr__(self, "bytes", calculated_bytes)

    @property
    def is_n_dependent(self) -> bool:
        return self.n_dependency == _N_DEPENDENT_ALLOWLIST

    def bytes_at_sequence_length(self, sequence_length: int, *, base_n: int) -> int:
        """Evaluate this concrete allocation at another sequence length."""
        _require_positive(sequence_length=sequence_length, base_n=base_n)
        scaled_shape = list(self.shape)
        for index, axis in enumerate(self.axes):
            if axis == "N":
                if scaled_shape[index] % base_n:
                    raise ValueError(
                        f"allocation {self.name} has a non-integral N scaling formula"
                    )
                scaled_shape[index] = scaled_shape[index] // base_n * sequence_length
        return prod(scaled_shape) * _DTYPE_BYTES[self.dtype]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "axes": list(self.axes),
            "bytes": self.bytes,
            "lifetime": list(self.lifetime),
            "n_dependency": self.n_dependency,
            "alias_of": self.alias_of,
            "formula": self.formula,
        }

    def to_json(self) -> str:
        return _stable_json(self.to_dict())


@dataclass(frozen=True)
class WorkspacePlan:
    """Allocation-level, liveness-checked workspace contract for one plan."""

    sequence_length: int
    workspace_budget_bytes: int
    allocations: tuple[WorkspaceAllocation, ...]
    autograd_saved_tensors: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_positive(sequence_length=self.sequence_length)
        allocations = tuple(self.allocations)
        autograd_saved_tensors = tuple(self.autograd_saved_tensors)
        if any(
            not isinstance(allocation, WorkspaceAllocation)
            for allocation in allocations
        ):
            raise TypeError("workspace allocations must be WorkspaceAllocation records")
        if any(
            not isinstance(saved_name, str) or not saved_name
            for saved_name in autograd_saved_tensors
        ):
            raise TypeError("autograd saved tensor names must be non-empty strings")
        object.__setattr__(self, "allocations", allocations)
        object.__setattr__(self, "autograd_saved_tensors", autograd_saved_tensors)
        if (
            not isinstance(self.workspace_budget_bytes, int)
            or isinstance(self.workspace_budget_bytes, bool)
            or self.workspace_budget_bytes <= 0
        ):
            raise ValueError("workspace budget must be a positive byte count")
        names = [allocation.name for allocation in self.allocations]
        if len(names) != len(set(names)):
            raise ValueError("workspace allocation names must be unique")
        known_names = set(names) | set(_EXTERNAL_ALIAS_TARGETS)
        for allocation in self.allocations:
            axes = set(allocation.axes)
            if (
                allocation.alias_of is not None
                and allocation.alias_of not in known_names
            ):
                raise ValueError(
                    f"workspace allocation {allocation.name} aliases an unknown tensor"
                )
            if allocation.name == "forward_pair_partial" and (
                allocation.dtype != "float32"
                or allocation.axes != ("B", "Hq", "pair_groups", "macro_token", "Caug")
                or allocation.lifetime != (1, 2)
                or allocation.n_dependency != _N_INDEPENDENT
                or allocation.alias_of is not None
                or allocation.formula
                != "4 * B * Hq * ceil(P / pair_block) * BT * (DV + 1)"
            ):
                raise ValueError(
                    "forward_pair_partial must use the bounded macro all-pair contract"
                )
            if ({"N", "chunks"} & axes) and ({"P", "Caug"} & axes):
                if not (
                    allocation.name == "normalization_g"
                    and allocation.axes == ("B", "Hq", "N", "Caug")
                ):
                    raise ValueError(
                        "workspace forbids a token/chunk by pair or Caug allocation"
                    )
            if allocation.n_dependency == _N_DEPENDENT_ALLOWLIST:
                if allocation.name not in _N_DEPENDENT_ALLOWLIST_NAMES:
                    raise ValueError(
                        f"N-dependent allocation {allocation.name} is not on the allowlist"
                    )
                if "N" not in axes:
                    raise ValueError(
                        f"N-dependent allocation {allocation.name} must expose its N axis"
                    )
            elif "N" in axes or "chunks" in axes:
                raise ValueError(
                    f"N-independent allocation {allocation.name} cannot contain N or chunks"
                )
            if "chunks" in axes:
                raise ValueError(
                    "workspace never permits a chunk-count allocation axis"
                )
            if "replicas" in axes:
                replica_count = allocation.shape[allocation.axes.index("replicas")]
                if replica_count not in _SUPPORTED_REDUCTION_REPLICAS:
                    raise ValueError(
                        "workspace reduction replica count must be bounded"
                    )
        name_to_allocation = {
            allocation.name: allocation for allocation in self.allocations
        }
        for metadata_name, rows_name, columns_name in _PAIR_METADATA_CONTRACTS:
            contract_names = {metadata_name, rows_name, columns_name}
            present_names = contract_names & set(name_to_allocation)
            if not present_names:
                continue
            if present_names != contract_names:
                raise ValueError(
                    f"{metadata_name} must be one physical record with row/column aliases"
                )
            metadata = name_to_allocation[metadata_name]
            if (
                metadata.dtype != "int32"
                or metadata.shape[0] != 2
                or metadata.axes != ("pair_component", "P")
                or metadata.alias_of is not None
                or metadata.n_dependency != _N_INDEPENDENT
                or metadata.formula != "4 * 2 * P"
            ):
                raise ValueError(
                    f"{metadata_name} must be physical int32 (2, P) metadata"
                )
            for alias_name in (rows_name, columns_name):
                alias = name_to_allocation[alias_name]
                if (
                    alias.dtype != "int32"
                    or alias.shape != (metadata.shape[1],)
                    or alias.axes != ("P",)
                    or alias.alias_of != metadata_name
                    or alias.lifetime != metadata.lifetime
                    or alias.n_dependency != _N_INDEPENDENT
                    or alias.formula != f"alias_of({metadata_name}); 4 * P"
                ):
                    raise ValueError(
                        f"{alias_name} must be an exact {metadata_name} row alias"
                    )
        for allocation in self.allocations:
            if allocation.alias_of is None:
                continue
            seen_aliases = {allocation.name}
            alias_target = allocation.alias_of
            while alias_target not in _EXTERNAL_ALIAS_TARGETS:
                if alias_target in seen_aliases:
                    raise ValueError(
                        f"workspace alias cycle detected for {allocation.name}"
                    )
                seen_aliases.add(alias_target)
                target_allocation = name_to_allocation.get(alias_target)
                if target_allocation is None:
                    raise ValueError(
                        f"workspace allocation {allocation.name} aliases an unknown tensor"
                    )
                if target_allocation.alias_of is None:
                    break
                alias_target = target_allocation.alias_of
        for saved_name in self.autograd_saved_tensors:
            allocation = name_to_allocation.get(saved_name)
            if allocation is None:
                raise ValueError(
                    f"autograd saved tensor {saved_name} is not listed in the workspace"
                )
            if allocation.alias_of is None:
                # A non-alias record is an explicit, fully-accounted allocation.
                continue
        if self.peak_n_independent_bytes > self.workspace_budget_bytes:
            raise ValueError(
                "workspace budget is insufficient for simultaneously-live "
                "N-independent allocations"
            )

    @property
    def n_dependent_allowlist_names(self) -> frozenset[str]:
        return _N_DEPENDENT_ALLOWLIST_NAMES

    @property
    def n_dependent_allocations(self) -> tuple[WorkspaceAllocation, ...]:
        return tuple(
            allocation
            for allocation in self.allocations
            if allocation.n_dependency == _N_DEPENDENT_ALLOWLIST
        )

    @property
    def peak_n_independent_bytes(self) -> int:
        physical_allocations = tuple(
            allocation
            for allocation in self.allocations
            if allocation.n_dependency == _N_INDEPENDENT and allocation.alias_of is None
        )
        if not physical_allocations:
            return 0
        endpoints = sorted(
            {
                endpoint
                for allocation in physical_allocations
                for endpoint in allocation.lifetime
            }
        )
        return max(
            sum(
                allocation.bytes
                for allocation in physical_allocations
                if allocation.lifetime[0] <= point <= allocation.lifetime[1]
            )
            for point in endpoints
        )

    def allocation(self, name: str) -> WorkspaceAllocation:
        for allocation in self.allocations:
            if allocation.name == name:
                return allocation
        raise KeyError(name)

    def allocation_by_name(self, name: str) -> WorkspaceAllocation:
        """Compatibility-friendly spelling for audit consumers."""
        return self.allocation(name)

    def bytes_at_sequence_length(self, sequence_length: int) -> dict[str, int]:
        return {
            allocation.name: allocation.bytes_at_sequence_length(
                sequence_length,
                base_n=self.sequence_length,
            )
            for allocation in self.allocations
        }

    def n_scaling_audit(self) -> dict[str, tuple[int, int, int]]:
        """Return exact per-record bytes at N, 2N, and 4N."""
        sequence_lengths = (
            self.sequence_length,
            2 * self.sequence_length,
            4 * self.sequence_length,
        )
        evaluated = tuple(
            self.bytes_at_sequence_length(sequence_length)
            for sequence_length in sequence_lengths
        )
        return {
            allocation.name: tuple(point[allocation.name] for point in evaluated)
            for allocation in self.allocations
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_length": self.sequence_length,
            "workspace_budget_bytes": self.workspace_budget_bytes,
            "peak_n_independent_bytes": self.peak_n_independent_bytes,
            "allocations": [allocation.to_dict() for allocation in self.allocations],
            "autograd_saved_tensors": list(self.autograd_saved_tensors),
        }

    def to_json(self) -> str:
        return _stable_json(self.to_dict())


def _freeze_axis_mapping(
    mapping: Mapping[str, int | bool | str],
    *,
    name: str,
) -> Mapping[str, int | bool | str]:
    if not isinstance(mapping, Mapping):
        raise TypeError(f"{name} must be a mapping")
    frozen: dict[str, int | bool | str] = {}
    for key, value in mapping.items():
        if not isinstance(key, str) or not key:
            raise TypeError(f"{name} keys must be non-empty strings")
        if not isinstance(value, (int, bool, str)):
            raise TypeError(f"{name} axis values must be JSON scalar values")
        frozen[key] = value
    return MappingProxyType(frozen)


def _freeze_string_mapping(
    mapping: Mapping[str, str], *, name: str
) -> Mapping[str, str]:
    if not isinstance(mapping, Mapping):
        raise TypeError(f"{name} must be a mapping")
    frozen: dict[str, str] = {}
    for key, value in mapping.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
        ):
            raise TypeError(f"{name} keys and values must be non-empty strings")
        frozen[key] = value
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class KernelPlan:
    """Immutable deterministic planner result; this is metadata, not dispatch."""

    geometry: KernelGeometry
    workspace: WorkspacePlan
    physical_path: str
    planner_version: str
    planner_source_hash: str
    specialization_axes: Mapping[str, int | bool | str]
    runtime_axes: Mapping[str, int | bool | str]
    canonical_pair_order: Mapping[str, str]
    plan_id: str = ""

    def __post_init__(self) -> None:
        if self.physical_path != PHYSICAL_PATH_IDENTIFIER:
            raise ValueError(
                "kernel plan physical path must be the canonical identifier"
            )
        if not self.planner_version or not self.planner_source_hash:
            raise ValueError("kernel plan must record planner version and source hash")
        specialization_axes = _freeze_axis_mapping(
            self.specialization_axes,
            name="specialization_axes",
        )
        runtime_axes = _freeze_axis_mapping(self.runtime_axes, name="runtime_axes")
        canonical_pair_order = _freeze_string_mapping(
            self.canonical_pair_order,
            name="canonical_pair_order",
        )
        required_pair_order = {
            "kind": "lower_triangular_row_major",
            "formula": "p(r,s)=r*(r+1)//2+s",
        }
        if not required_pair_order.items() <= canonical_pair_order.items():
            raise ValueError("kernel plan must record canonical packed pair order")
        object.__setattr__(self, "specialization_axes", specialization_axes)
        object.__setattr__(self, "runtime_axes", runtime_axes)
        object.__setattr__(self, "canonical_pair_order", canonical_pair_order)
        expected_plan_id = hashlib.sha256(
            _stable_json(self.payload_without_plan_id()).encode("utf-8")
        ).hexdigest()
        if self.plan_id and self.plan_id != expected_plan_id:
            raise ValueError("kernel plan id does not match its stable payload")
        object.__setattr__(self, "plan_id", expected_plan_id)

    def payload_without_plan_id(self) -> dict[str, Any]:
        return {
            "geometry": self.geometry.to_dict(),
            "workspace": self.workspace.to_dict(),
            "physical_path": self.physical_path,
            "planner_version": self.planner_version,
            "planner_source_hash": self.planner_source_hash,
            "specialization_axes": dict(self.specialization_axes),
            "runtime_axes": dict(self.runtime_axes),
            "canonical_pair_order": dict(self.canonical_pair_order),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload_without_plan_id(), "plan_id": self.plan_id}

    def to_json(self) -> str:
        return _stable_json(self.to_dict())


def _workspace_allocation(
    name: str,
    dtype: str,
    shape: tuple[int, ...],
    axes: tuple[str, ...],
    lifetime: tuple[int, int] | str,
    n_dependency: str,
    *,
    alias_of: str | None = None,
    formula: str = "",
) -> WorkspaceAllocation:
    return WorkspaceAllocation(
        name=name,
        dtype=dtype,
        shape=shape,
        axes=axes,
        lifetime=lifetime,
        n_dependency=n_dependency,
        alias_of=alias_of,
        formula=formula,
    )


def _build_workspace_plan(
    geometry: KernelGeometry,
    *,
    macro_token_block: int,
    pair_block: int,
    value_block: int,
    reduction_replicas: int,
) -> WorkspacePlan:
    """Build concrete allocations without looking at a device allocator."""
    if geometry.workspace_budget_bytes is None:
        raise ValueError("kernel geometry must carry a resolved workspace budget")
    if geometry.launch_stage == "normalization_vjp":
        return _build_normalization_vjp_workspace_plan(
            geometry,
            value_block=value_block,
        )
    B, Hq, Hkv, N, D, DV, Gmax = (
        geometry.B,
        geometry.Hq,
        geometry.Hkv,
        geometry.N,
        geometry.D,
        geometry.DV,
        geometry.gmax,
    )
    P = geometry.pair_count
    Caug = geometry.augmented_value_dimension
    if pair_block not in _SUPPORTED_PAIR_BLOCKS:
        raise ValueError("forward pair_block must be a bounded supported schedule")
    if macro_token_block not in _SUPPORTED_TOKEN_BLOCKS:
        raise ValueError(
            "forward macro_token_block must be a bounded supported schedule"
        )
    if value_block not in _SUPPORTED_VALUE_BLOCKS:
        raise ValueError("value_block must be a bounded supported schedule")
    if reduction_replicas not in _SUPPORTED_GROUPS_PER_WAVE:
        raise ValueError("groups_per_wave must be a bounded supported schedule")
    pair_groups = (P + pair_block - 1) // pair_block
    if not 0 < pair_groups <= P:
        raise ValueError("forward pair_groups must be a bounded packed-pair partition")
    if not (P <= pair_groups * pair_block and P > (pair_groups - 1) * pair_block):
        raise AssertionError("forward pair_groups must equal ceil(P / pair_block)")
    data_dtype = geometry.dtype
    value_blocks = (Caug + value_block - 1) // value_block
    output_axes = ("B", "Hq", "N", "DV")
    denominator_axes = ("B", "Hq", "N", "one")
    q_axes = ("B", "Hq", "N", "D")
    kv_axes = ("B", "Hkv", "N", "D")
    v_axes = ("B", "Hkv", "N", "DV")
    allocations: list[WorkspaceAllocation] = [
        _workspace_allocation(
            "q", data_dtype, (B, Hq, N, D), q_axes, "forward", _N_DEPENDENT_ALLOWLIST
        ),
        _workspace_allocation(
            "k", data_dtype, (B, Hkv, N, D), kv_axes, "forward", _N_DEPENDENT_ALLOWLIST
        ),
        _workspace_allocation(
            "v", data_dtype, (B, Hkv, N, DV), v_axes, "forward", _N_DEPENDENT_ALLOWLIST
        ),
        _workspace_allocation(
            "output",
            data_dtype,
            (B, Hq, N, DV),
            output_axes,
            "forward",
            _N_DEPENDENT_ALLOWLIST,
        ),
        _workspace_allocation(
            "numerator",
            "float32",
            (B, Hq, N, DV),
            output_axes,
            "forward",
            _N_DEPENDENT_ALLOWLIST,
        ),
        _workspace_allocation(
            "denominator",
            "float32",
            (B, Hq, N, 1),
            denominator_axes,
            "forward",
            _N_DEPENDENT_ALLOWLIST,
        ),
        _workspace_allocation(
            "d_output",
            data_dtype,
            (B, Hq, N, DV),
            output_axes,
            "normalization",
            _N_DEPENDENT_ALLOWLIST,
        ),
        _workspace_allocation(
            "d_numerator",
            "float32",
            (B, Hq, N, DV),
            output_axes,
            "normalization",
            _N_DEPENDENT_ALLOWLIST,
        ),
        _workspace_allocation(
            "d_denominator",
            "float32",
            (B, Hq, N, 1),
            denominator_axes,
            "normalization",
            _N_DEPENDENT_ALLOWLIST,
        ),
        _workspace_allocation(
            "dq", "float32", (B, Hq, N, D), q_axes, "prefix", _N_DEPENDENT_ALLOWLIST
        ),
        _workspace_allocation(
            "normalization_g",
            "float32",
            (B, Hq, N, Caug),
            ("B", "Hq", "N", "Caug"),
            "normalization",
            _N_DEPENDENT_ALLOWLIST,
        ),
        _workspace_allocation(
            "denominator_partial",
            "float32",
            (B, Hq, N),
            ("B", "Hq", "N"),
            "forward",
            _N_DEPENDENT_ALLOWLIST,
            formula="4 * B * Hq * N",
        ),
        _workspace_allocation(
            "forward_state",
            "float32",
            (B, Hkv, Caug, 1 + D + P),
            ("B", "Hkv", "Caug", "state_features"),
            (0, 1),
            _N_INDEPENDENT,
            formula="4 * B * Hkv * (DV + 1) * (1 + D + D * (D + 1) // 2)",
        ),
        _workspace_allocation(
            "forward_pair_partial",
            "float32",
            (B, Hq, pair_groups, macro_token_block, Caug),
            ("B", "Hq", "pair_groups", "macro_token", "Caug"),
            (1, 2),
            _N_INDEPENDENT,
            formula="4 * B * Hq * ceil(P / pair_block) * BT * (DV + 1)",
        ),
    ]
    needs_q, needs_k, needs_v, needs_a, needs_b, needs_c = (
        geometry.requested_gradient_mask
    )
    if needs_q or needs_a or needs_b or needs_c:
        allocations.extend(
            (
                _workspace_allocation(
                    "prefix_state",
                    "float32",
                    (B, Hkv, Caug, 1 + D + P),
                    ("B", "Hkv", "Caug", "state_features"),
                    (5, 7),
                    _N_INDEPENDENT,
                    formula="4 * B * Hkv * (DV + 1) * (1 + D + D * (D + 1) // 2)",
                ),
                _workspace_allocation(
                    "prefix_h0_view",
                    "float32",
                    (B, Hkv, Caug),
                    ("B", "Hkv", "Caug"),
                    (5, 7),
                    _N_INDEPENDENT,
                    alias_of="prefix_state",
                ),
                _workspace_allocation(
                    "prefix_h1_view",
                    "float32",
                    (B, Hkv, D, Caug),
                    ("B", "Hkv", "D", "Caug"),
                    (5, 7),
                    _N_INDEPENDENT,
                    alias_of="prefix_state",
                ),
                _workspace_allocation(
                    "prefix_h2_view",
                    "float32",
                    (B, Hkv, P, Caug),
                    ("B", "Hkv", "P", "Caug"),
                    (5, 7),
                    _N_INDEPENDENT,
                    alias_of="prefix_state",
                ),
            )
        )
        if needs_q or needs_c:
            allocations.extend(
                (
                    _workspace_allocation(
                        "prefix_pair_metadata",
                        "int32",
                        (2, P),
                        ("pair_component", "P"),
                        (5, 7),
                        _N_INDEPENDENT,
                        formula="4 * 2 * P",
                    ),
                    _workspace_allocation(
                        "prefix_pair_rows",
                        "int32",
                        (P,),
                        ("P",),
                        (5, 7),
                        _N_INDEPENDENT,
                        alias_of="prefix_pair_metadata",
                        formula="4 * P",
                    ),
                    _workspace_allocation(
                        "prefix_pair_columns",
                        "int32",
                        (P,),
                        ("P",),
                        (5, 7),
                        _N_INDEPENDENT,
                        alias_of="prefix_pair_metadata",
                        formula="4 * P",
                    ),
                    _workspace_allocation(
                        "prefix_h2_dot_partial",
                        "float32",
                        (
                            B,
                            Hq,
                            pair_groups,
                            macro_token_block,
                            value_blocks,
                            pair_block,
                        ),
                        ("B", "Hq", "pair_groups", "BT", "value_blocks", "BP"),
                        (5, 6),
                        _N_INDEPENDENT,
                        formula="4 * B * Hq * ceil(P / BP) * BT * ceil(Caug / BV) * BP",
                    ),
                )
            )
        if needs_a:
            allocations.extend(
                (
                    _workspace_allocation(
                        "prefix_h0_dot_partial",
                        "float32",
                        (B, Hq, macro_token_block, value_blocks),
                        ("B", "Hq", "BT", "value_blocks"),
                        (5, 6),
                        _N_INDEPENDENT,
                        formula="4 * B * Hq * BT * ceil(Caug / BV)",
                    ),
                    _workspace_allocation(
                        "d_a_macro",
                        "float32",
                        (B, Hq, macro_token_block),
                        ("B", "Hq", "BT"),
                        (6, 7),
                        _N_INDEPENDENT,
                        formula="4 * B * Hq * BT",
                    ),
                )
            )
        if needs_q or needs_b:
            allocations.append(
                _workspace_allocation(
                    "prefix_h1_dot_partial",
                    "float32",
                    (B, Hq, macro_token_block, value_blocks, D),
                    ("B", "Hq", "BT", "value_blocks", "D"),
                    (5, 6),
                    _N_INDEPENDENT,
                    formula="4 * B * Hq * BT * ceil(Caug / BV) * D",
                )
            )
        if needs_q:
            allocations.extend(
                (
                    _workspace_allocation(
                        "prefix_dq_macro",
                        "float32",
                        (B, Hq, macro_token_block, D),
                        ("B", "Hq", "BT", "D"),
                        (6, 7),
                        _N_INDEPENDENT,
                        formula="4 * B * Hq * BT * D",
                    ),
                )
            )
        if needs_c:
            allocations.extend(
                (
                    _workspace_allocation(
                        "d_c_macro",
                        "float32",
                        (B, Hq, macro_token_block, Gmax, Gmax),
                        ("B", "Hq", "BT", "Gmax", "Gmax"),
                        (6, 7),
                        _N_INDEPENDENT,
                        formula="4 * B * Hq * BT * Gmax * Gmax",
                    ),
                )
            )
        if needs_b:
            allocations.append(
                _workspace_allocation(
                    "d_b_macro",
                    "float32",
                    (B, Hq, macro_token_block, Gmax),
                    ("B", "Hq", "BT", "Gmax"),
                    (6, 7),
                    _N_INDEPENDENT,
                    formula="4 * B * Hq * BT * Gmax",
                )
            )
    if needs_k or needs_v:
        allocations.extend(
            (
                _workspace_allocation(
                    "suffix_state",
                    "float32",
                    (B, Hkv, Caug, 1 + D + P),
                    ("B", "Hkv", "Caug", "state_features"),
                    (8, 10),
                    _N_INDEPENDENT,
                    formula="4 * B * Hkv * (DV + 1) * (1 + D + D * (D + 1) // 2)",
                ),
                _workspace_allocation(
                    "suffix_h0_view",
                    "float32",
                    (B, Hkv, Caug),
                    ("B", "Hkv", "Caug"),
                    (8, 10),
                    _N_INDEPENDENT,
                    alias_of="suffix_state",
                ),
                _workspace_allocation(
                    "suffix_h1_view",
                    "float32",
                    (B, Hkv, D, Caug),
                    ("B", "Hkv", "D", "Caug"),
                    (8, 10),
                    _N_INDEPENDENT,
                    alias_of="suffix_state",
                ),
                _workspace_allocation(
                    "suffix_h2_view",
                    "float32",
                    (B, Hkv, P, Caug),
                    ("B", "Hkv", "P", "Caug"),
                    (8, 10),
                    _N_INDEPENDENT,
                    alias_of="suffix_state",
                ),
                _workspace_allocation(
                    "suffix_pair_metadata",
                    "int32",
                    (2, P),
                    ("pair_component", "P"),
                    (8, 10),
                    _N_INDEPENDENT,
                    formula="4 * 2 * P",
                ),
                _workspace_allocation(
                    "suffix_pair_rows",
                    "int32",
                    (P,),
                    ("P",),
                    (8, 10),
                    _N_INDEPENDENT,
                    alias_of="suffix_pair_metadata",
                    formula="4 * P",
                ),
                _workspace_allocation(
                    "suffix_pair_columns",
                    "int32",
                    (P,),
                    ("P",),
                    (8, 10),
                    _N_INDEPENDENT,
                    alias_of="suffix_pair_metadata",
                    formula="4 * P",
                ),
            )
        )
        if needs_v:
            allocations.extend(
                (
                    _workspace_allocation(
                        "dv",
                        "float32",
                        (B, Hkv, N, DV),
                        v_axes,
                        "suffix",
                        _N_DEPENDENT_ALLOWLIST,
                    ),
                    _workspace_allocation(
                        "suffix_dv_macro",
                        "float32",
                        (B, Hkv, macro_token_block, DV),
                        ("B", "Hkv", "BT", "DV"),
                        (8, 10),
                        _N_INDEPENDENT,
                        formula="4 * B * Hkv * BT * DV",
                    ),
                    _workspace_allocation(
                        "suffix_dv_pair_partial",
                        "float32",
                        (B, Hkv, pair_groups, macro_token_block, Caug),
                        ("B", "Hkv", "pair_groups", "BT", "Caug"),
                        (8, 10),
                        _N_INDEPENDENT,
                        formula="4 * B * Hkv * ceil(P / BP) * BT * Caug",
                    ),
                )
            )
        if needs_k:
            allocations.extend(
                (
                    _workspace_allocation(
                        "dk",
                        "float32",
                        (B, Hkv, N, D),
                        kv_axes,
                        "suffix",
                        _N_DEPENDENT_ALLOWLIST,
                    ),
                    _workspace_allocation(
                        "suffix_dk_value_partial",
                        "float32",
                        (B, Hkv, macro_token_block, value_blocks, D),
                        ("B", "Hkv", "BT", "value_blocks", "D"),
                        (8, 10),
                        _N_INDEPENDENT,
                        formula="4 * B * Hkv * BT * ceil(Caug / BV) * D",
                    ),
                    _workspace_allocation(
                        "suffix_dk_pair_partial",
                        "float32",
                        (B, Hkv, pair_groups, macro_token_block, value_blocks, D),
                        ("B", "Hkv", "pair_groups", "BT", "value_blocks", "D"),
                        (8, 10),
                        _N_INDEPENDENT,
                        formula="4 * B * Hkv * ceil(P / BP) * BT * ceil(Caug / BV) * D",
                    ),
                )
            )
    if needs_a:
        allocations.append(
            _workspace_allocation(
                "d_constant", "float32", (Hq,), ("Hq",), (6, 7), _N_INDEPENDENT
            )
        )
    if needs_b:
        allocations.append(
            _workspace_allocation(
                "d_linear",
                "float32",
                (Hq, Gmax),
                ("Hq", "Gmax"),
                (6, 7),
                _N_INDEPENDENT,
            )
        )
    if needs_c:
        allocations.append(
            _workspace_allocation(
                "d_quadratic",
                "float32",
                (Hq, Gmax, Gmax),
                ("Hq", "Gmax", "Gmax"),
                (6, 7),
                _N_INDEPENDENT,
            )
        )
    dim_group_shape = (D,) if geometry.group_layout == "shared" else (Hq, D)
    dim_group_axes = ("D",) if geometry.group_layout == "shared" else ("Hq", "D")
    allocations.extend(
        (
            _workspace_allocation(
                "saved_q",
                data_dtype,
                (B, Hq, N, D),
                q_axes,
                "saved",
                _N_DEPENDENT_ALLOWLIST,
                alias_of="q",
            ),
            _workspace_allocation(
                "saved_k",
                data_dtype,
                (B, Hkv, N, D),
                kv_axes,
                "saved",
                _N_DEPENDENT_ALLOWLIST,
                alias_of="k",
            ),
            _workspace_allocation(
                "saved_v",
                data_dtype,
                (B, Hkv, N, DV),
                v_axes,
                "saved",
                _N_DEPENDENT_ALLOWLIST,
                alias_of="v",
            ),
            _workspace_allocation(
                "saved_dim_groups",
                "int32",
                dim_group_shape,
                dim_group_axes,
                "saved",
                _N_INDEPENDENT,
                alias_of="dim_groups",
            ),
            _workspace_allocation(
                "saved_constant",
                "float32",
                (Hq,),
                ("Hq",),
                "saved",
                _N_INDEPENDENT,
                alias_of="constant",
            ),
            _workspace_allocation(
                "saved_linear",
                "float32",
                (Hq, Gmax),
                ("Hq", "Gmax"),
                "saved",
                _N_INDEPENDENT,
                alias_of="linear",
            ),
            _workspace_allocation(
                "saved_quadratic",
                "float32",
                (Hq, Gmax, Gmax),
                ("Hq", "Gmax", "Gmax"),
                "saved",
                _N_INDEPENDENT,
                alias_of="quadratic",
            ),
            _workspace_allocation(
                "saved_numerator",
                "float32",
                (B, Hq, N, DV),
                output_axes,
                "saved",
                _N_DEPENDENT_ALLOWLIST,
                alias_of="numerator",
            ),
            _workspace_allocation(
                "saved_denominator",
                "float32",
                (B, Hq, N, 1),
                denominator_axes,
                "saved",
                _N_DEPENDENT_ALLOWLIST,
                alias_of="denominator",
            ),
        )
    )
    return WorkspacePlan(
        sequence_length=N,
        workspace_budget_bytes=geometry.workspace_budget_bytes,
        allocations=tuple(allocations),
        autograd_saved_tensors=(
            "saved_q",
            "saved_k",
            "saved_v",
            "saved_dim_groups",
            "saved_constant",
            "saved_linear",
            "saved_quadratic",
            "saved_numerator",
            "saved_denominator",
        ),
    )


def _build_normalization_vjp_workspace_plan(
    geometry: KernelGeometry,
    *,
    value_block: int,
) -> WorkspacePlan:
    """Account for the exact FP32 scratch of ``augmented_normalization_vjp``.

    This intentionally excludes H0/H1/H2 and all prefix/suffix state.  The
    actual Triton normalization entry materializes a contiguous augmented G,
    a fixed value-block reduction buffer, and (at most) one FP32 view/copy for
    each incoming public cotangent.  The numerator and denominator are saved
    forward outputs, represented here only as external aliases.
    """
    if geometry.workspace_budget_bytes is None:
        raise ValueError("normalization workspace requires a resolved budget")
    B, Hq, N, DV = geometry.B, geometry.Hq, geometry.N, geometry.DV
    Caug = geometry.augmented_value_dimension
    value_blocks = (DV + value_block - 1) // value_block
    output_axes = ("B", "Hq", "N", "DV")
    denominator_axes = ("B", "Hq", "N", "one")
    allocations = (
        _workspace_allocation(
            "normalization_numerator",
            "float32",
            (B, Hq, N, DV),
            output_axes,
            "normalization",
            _N_DEPENDENT_ALLOWLIST,
            alias_of="numerator",
        ),
        _workspace_allocation(
            "normalization_denominator",
            "float32",
            (B, Hq, N, 1),
            denominator_axes,
            "normalization",
            _N_DEPENDENT_ALLOWLIST,
            alias_of="denominator",
        ),
        _workspace_allocation(
            "normalization_grad_output",
            "float32",
            (B, Hq, N, DV),
            output_axes,
            "normalization",
            _N_DEPENDENT_ALLOWLIST,
            formula="4 * B * Hq * N * DV; maximum optional grad_output conversion",
        ),
        _workspace_allocation(
            "normalization_grad_numerator",
            "float32",
            (B, Hq, N, DV),
            output_axes,
            "normalization",
            _N_DEPENDENT_ALLOWLIST,
            formula="4 * B * Hq * N * DV; maximum optional grad_numerator conversion",
        ),
        _workspace_allocation(
            "normalization_grad_denominator",
            "float32",
            (B, Hq, N, 1),
            denominator_axes,
            "normalization",
            _N_DEPENDENT_ALLOWLIST,
            formula="4 * B * Hq * N; maximum optional grad_denominator conversion",
        ),
        _workspace_allocation(
            "normalization_g",
            "float32",
            (B, Hq, N, Caug),
            ("B", "Hq", "N", "Caug"),
            "normalization",
            _N_DEPENDENT_ALLOWLIST,
            formula="4 * B * Hq * N * (DV + 1); contiguous augmented G",
        ),
        _workspace_allocation(
            "normalization_denominator_partials",
            "float32",
            (B, Hq, N, value_blocks),
            ("B", "Hq", "N", "value_blocks"),
            "normalization",
            _N_DEPENDENT_ALLOWLIST,
            formula="4 * B * Hq * N * ceil(DV / BV); normalization value-block reduction",
        ),
    )
    return WorkspacePlan(
        sequence_length=N,
        workspace_budget_bytes=geometry.workspace_budget_bytes,
        allocations=allocations,
        autograd_saved_tensors=(),
    )


def _bt32_static_rejection_reason(geometry: KernelGeometry) -> str | None:
    """Return an auditable generic reason BT=32 cannot be considered.

    These predicates describe the bounded Triton program shape and the declared
    CUDA resource tuple.  They intentionally use no product name, UUID, total
    VRAM, allocator state, or benchmark result; the caller separately tests
    the exact candidate workspace against the already-resolved budget.
    """
    if geometry.sequence_length % 32:
        return "bt32_rejected_sequence_tail"
    mask = geometry.requested_gradient_mask
    # A public all-gradient plan is split before VJP launch into these two
    # complete stage masks.  They are the actual prefix and suffix workloads,
    # so BT=32 must evaluate them as candidates too.  A partial stage (for
    # example Q-only) stays at BT=16: its short launch work cannot justify the
    # larger bounded program.  This is a mask/work criterion, never a device
    # model rule.
    prefix_mask = (True, False, False, True, True, True)
    suffix_mask = (False, True, True, False, False, False)
    if mask not in ((True, True, True, True, True, True), prefix_mask, suffix_mask):
        return "bt32_rejected_sparse_gradient_mask"
    if geometry.pair_count < 1024 or geometry.augmented_value_dimension < 32:
        return "bt32_rejected_insufficient_pair_work"
    if geometry.compute_capability[0] < 8:
        return "bt32_rejected_compute_capability"
    if geometry.max_threads_per_block < 1024:
        return "bt32_rejected_max_threads_per_block"
    if geometry.max_threads_per_sm < 1024:
        return "bt32_rejected_max_threads_per_sm"
    if geometry.registers_per_block < 65_536:
        return "bt32_rejected_registers_per_block"
    if geometry.registers_per_sm < 65_536:
        return "bt32_rejected_registers_per_sm"
    if geometry.shared_memory_per_block_optin < 64 * 2**10:
        return "bt32_rejected_shared_memory_per_block"
    if geometry.shared_memory_per_sm < 64 * 2**10:
        return "bt32_rejected_shared_memory_per_sm"
    return None


def _select_bounded_schedule(
    geometry: KernelGeometry,
) -> tuple[int, int, int, int, str]:
    """Select a finite schedule from geometry and exact candidate workspace.

    BT=32 is a controlled candidate, not a device-name rule.  When it is not
    admitted, BT=16 remains the exact fallback and the immutable plan records
    why, so launchers cannot silently substitute either schedule.
    """
    if geometry.launch_stage == "normalization_vjp":
        return (
            16,
            32,
            NORMALIZATION_VALUE_BLOCK,
            4,
            "bt16_normalization_fixed_schedule",
        )
    pair_block = next(
        block
        for block in sorted(_SUPPORTED_PAIR_BLOCKS)
        if block >= min(geometry.pair_count, 128)
    )
    value_block = next(
        block
        for block in sorted(_SUPPORTED_VALUE_BLOCKS)
        if block >= min(geometry.DV, 32)
    )
    reduction_replicas = 4
    rejection = _bt32_static_rejection_reason(geometry)
    if rejection is None:
        try:
            _build_workspace_plan(
                geometry,
                macro_token_block=32,
                pair_block=pair_block,
                value_block=value_block,
                reduction_replicas=reduction_replicas,
            )
        except ValueError as error:
            if "workspace budget is insufficient" not in str(error):
                raise
            rejection = "bt32_rejected_workspace_budget"
    if rejection is None:
        return 32, pair_block, value_block, reduction_replicas, "bt32_admitted"
    return 16, pair_block, value_block, reduction_replicas, rejection


def _resolved_workspace_budget(
    geometry: KernelGeometry,
    *,
    free_bytes: int | None,
    workspace_budget_bytes: int | None,
) -> int:
    """Compute/reject a budget from caller-provided free bytes only."""
    candidates: list[int] = []
    if free_bytes is not None:
        if (
            not isinstance(free_bytes, int)
            or isinstance(free_bytes, bool)
            or free_bytes < 0
        ):
            raise ValueError("free_bytes must be a non-negative integer")
        candidates.append(min(_MAX_AUXILIARY_WORKSPACE_BYTES, floor(0.05 * free_bytes)))
    for explicit_budget in (geometry.workspace_budget_bytes, workspace_budget_bytes):
        if explicit_budget is not None:
            if (
                not isinstance(explicit_budget, int)
                or isinstance(explicit_budget, bool)
                or explicit_budget <= 0
            ):
                raise ValueError("workspace budget must be a positive integer")
            if explicit_budget > _MAX_AUXILIARY_WORKSPACE_BYTES:
                raise ValueError(
                    "explicit workspace budget cannot exceed the hard 512 MiB cap"
                )
            candidates.append(explicit_budget)
    if not candidates:
        raise ValueError(
            "build_kernel_plan requires injected free_bytes or an explicit workspace budget"
        )
    budget = min(candidates)
    if budget <= 0:
        raise ValueError(
            "workspace budget is zero after applying the free-memory limit"
        )
    return budget


def build_kernel_plan(
    geometry: KernelGeometry,
    *,
    free_bytes: int | None = None,
    workspace_budget_bytes: int | None = None,
) -> KernelPlan:
    """Build a deterministic CPU-testable plan without inspecting CUDA state.

    The caller injects ``free_bytes`` for the dispatch-time budget calculation.
    This function neither queries allocator state nor routes any production
    kernel; later launch tasks consume this immutable record.
    """
    if not isinstance(geometry, KernelGeometry):
        raise TypeError("build_kernel_plan requires a KernelGeometry")
    resolved_geometry = replace(
        geometry,
        workspace_budget_bytes=_resolved_workspace_budget(
            geometry,
            free_bytes=free_bytes,
            workspace_budget_bytes=workspace_budget_bytes,
        ),
    )
    macro_token_block, pair_block, value_block, reduction_replicas, selection = (
        _select_bounded_schedule(resolved_geometry)
    )
    workspace = _build_workspace_plan(
        resolved_geometry,
        macro_token_block=macro_token_block,
        pair_block=pair_block,
        value_block=value_block,
        reduction_replicas=reduction_replicas,
    )
    return KernelPlan(
        geometry=resolved_geometry,
        workspace=workspace,
        physical_path=PHYSICAL_PATH_IDENTIFIER,
        planner_version=_DEFAULT_PLANNER_VERSION,
        planner_source_hash=_planner_source_hash(),
        specialization_axes={
            "macro_token_block": macro_token_block,
            "macro_token_block_selection": selection,
            "pair_block": pair_block,
            "value_block": value_block,
            "reduction_replicas": reduction_replicas,
            "groups_per_wave": reduction_replicas,
            "causal": resolved_geometry.causal,
            "dtype": resolved_geometry.dtype,
            "head_dimension": resolved_geometry.D,
            "value_dimension": resolved_geometry.DV,
            "gmax": resolved_geometry.gmax,
            "launch_stage": resolved_geometry.launch_stage,
        },
        runtime_axes={
            "batch_size": resolved_geometry.B,
            "query_heads": resolved_geometry.Hq,
            "key_value_heads": resolved_geometry.Hkv,
            "sequence_length": resolved_geometry.N,
            "pair_count": resolved_geometry.pair_count,
            "gqa_ratio": resolved_geometry.Hq // resolved_geometry.Hkv,
        },
        canonical_pair_order={
            "kind": "lower_triangular_row_major",
            "formula": "p(r,s)=r*(r+1)//2+s",
        },
    )


def _require_positive(**dimensions: int) -> None:
    if any(value <= 0 for value in dimensions.values()):
        raise ValueError("all grouped causal dimensions must be positive")


@dataclass(frozen=True)
class GroupedCausalScanConfig:
    """Bounded scheduling parameters shared by causal scan launchers."""

    token_block: int
    pair_block: int
    value_block: int
    groups_per_wave: int

    def __post_init__(self) -> None:
        _require_positive(
            token_block=self.token_block,
            pair_block=self.pair_block,
            value_block=self.value_block,
            groups_per_wave=self.groups_per_wave,
        )
        if self.token_block not in _SUPPORTED_TOKEN_BLOCKS:
            raise ValueError("token_block must be one of the supported schedules")
        if self.pair_block not in _SUPPORTED_PAIR_BLOCKS:
            raise ValueError("pair_block must be one of the supported schedules")
        if self.value_block not in _SUPPORTED_VALUE_BLOCKS:
            raise ValueError("value_block must be one of the supported schedules")
        if self.groups_per_wave not in _SUPPORTED_GROUPS_PER_WAVE:
            raise ValueError("groups_per_wave must be one of the supported schedules")


def planned_grouped_causal_scan_config(plan: KernelPlan) -> GroupedCausalScanConfig:
    """Validate and return the bounded schedule embedded in an exact plan.

    A launcher must use these axes verbatim after it has established the
    plan's complete live identity.  Replacing them with a legacy default
    schedule would make its workspace records disagree with the launched
    Triton blocks for small or large dimensions.
    """
    if not isinstance(plan, KernelPlan):
        raise TypeError("planned grouped causal schedule requires a KernelPlan")
    axes = plan.specialization_axes
    values = tuple(
        axes.get(name)
        for name in (
            "macro_token_block",
            "pair_block",
            "value_block",
            "groups_per_wave",
        )
    )
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise ValueError("KernelPlan omits an integer bounded causal schedule")
    return GroupedCausalScanConfig(*values)


@dataclass(frozen=True)
class GroupedCausalWorkspaceAccounting:
    """Logical workspace categories for one causal streaming scan."""

    state_bytes: int
    block_partial_bytes: int
    token_linear_bytes: int


@dataclass(frozen=True)
class GroupedCausalStateSnapshot:
    """Copied KV-owned H state at one inclusive causal token."""

    token_index: int
    h0: torch.Tensor
    h1: torch.Tensor
    h2: torch.Tensor


@dataclass(frozen=True)
class GroupedCausalForwardTokenWitness:
    """Diagnostic-only physical H update and same-token query contraction."""

    token_index: int
    macro_index: int
    h0_increment: torch.Tensor
    h1_increment: torch.Tensor
    h2_increment: torch.Tensor
    h0_inclusive: torch.Tensor
    h1_inclusive: torch.Tensor
    h2_inclusive: torch.Tensor
    constant_contraction: torch.Tensor
    linear_contraction: torch.Tensor
    quadratic_contraction: torch.Tensor
    pre_normalized_diagonal_contraction: torch.Tensor


@dataclass(frozen=True)
class GroupedCausalForwardDiagnosticWitness:
    """CPU-small-geometry forward inspection record; never a launch artifact."""

    physical_path: str
    pair_rows: torch.Tensor
    pair_columns: torch.Tensor
    pair_multiplicity: torch.Tensor
    query_to_key_value: torch.Tensor
    tokens: tuple[GroupedCausalForwardTokenWitness, ...]
    numerator: torch.Tensor
    denominator: torch.Tensor


@dataclass(frozen=True)
class GroupedCausalSuffixStateSnapshot:
    """Copied KV-owned reverse-inclusive A state for one causal token."""

    token_index: int
    a0: torch.Tensor
    a1: torch.Tensor
    a2: torch.Tensor


@dataclass(frozen=True)
class GroupedCausalBackwardDiagnosticWitness:
    """CPU-only normalization/prefix/suffix inspection record."""

    physical_path: str
    pair_rows: torch.Tensor
    pair_columns: torch.Tensor
    pair_multiplicity: torch.Tensor
    normalization_g: torch.Tensor
    prefix_h: tuple[GroupedCausalStateSnapshot, ...]
    suffix_a: tuple[GroupedCausalSuffixStateSnapshot, ...]


def canonical_packed_pair_metadata(
    head_dimension: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return canonical lower-triangular rows, columns, and multipliers."""
    _require_positive(head_dimension=head_dimension)
    pair_rows, pair_columns = torch.tril_indices(
        head_dimension, head_dimension, device=device, dtype=torch.int64
    )
    pair_multiplicity = torch.where(
        pair_rows == pair_columns,
        torch.ones(pair_rows.numel(), device=device, dtype=torch.float32),
        torch.full((pair_rows.numel(),), 2.0, device=device, dtype=torch.float32),
    )
    return pair_rows, pair_columns, pair_multiplicity


def grouped_causal_diagnostic_record_bytes(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> int:
    """Return the worst-case FP32 diagnostic record footprint before allocation.

    The forward record retains per-token increments and inclusive H states.
    A backward record additionally retains copied prefix H and reverse A
    states, so this intentionally charges the complete diagnostic route even
    when a caller asks for the forward witness alone.
    """
    batch_size, query_heads, token_count, head_dimension = q.shape
    key_value_heads = k.shape[1]
    value_dimension = v.shape[-1]
    augmented_dimension = value_dimension + 1
    pair_count = head_dimension * (head_dimension + 1) // 2
    state_elements = (
        batch_size
        * key_value_heads
        * augmented_dimension
        * (1 + head_dimension + pair_count)
    )
    query_record_elements = batch_size * query_heads * token_count * augmented_dimension
    # Six forward H tensors (increments plus inclusive states), then three
    # copied H-prefix and three reverse A snapshots in the backward witness.
    snapshot_elements = 12 * token_count * state_elements
    forward_contractions = 4 * query_record_elements
    forward_outputs = query_record_elements
    normalization_g = query_record_elements
    live_scan_state = 2 * state_elements
    # The diagnostic forward keeps all three GQA-expanded H views live before
    # contracting them.  The linear and packed-pair elementwise expressions
    # can coexist with those views, so charge a conservative peak rather than
    # counting only tensors retained by frozen witness records.
    expanded_query_state = (
        batch_size
        * query_heads
        * augmented_dimension
        * (1 + head_dimension + pair_count)
    )
    linear_contraction_work = (
        batch_size * query_heads * head_dimension * augmented_dimension
    )
    packed_pair_contraction_work = (
        batch_size * query_heads * pair_count * augmented_dimension
    )
    peak_query_work = (
        expanded_query_state
        + 2 * linear_contraction_work
        + 3 * packed_pair_contraction_work
    )
    pair_metadata_bytes = pair_count * (2 * 8 + _FP32_BYTES) + query_heads * 8
    return (
        _FP32_BYTES
        * (
            snapshot_elements
            + forward_contractions
            + forward_outputs
            + normalization_g
            + live_scan_state
            + peak_query_work
        )
        + pair_metadata_bytes
    )


def require_cpu_diagnostic_geometry(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """Reject any non-CPU or non-small diagnostic witness invocation.

    Diagnostic records are intentionally a tiny CPU oracle.  They are not
    saved by autograd, routed through a CUDA launch, or usable as production
    evidence.
    """
    if q.device.type != "cpu":
        raise ValueError("grouped causal diagnostic witness is CPU-only")
    if q.shape[2] > DIAGNOSTIC_SEQUENCE_CAP:
        raise ValueError(
            "grouped causal diagnostic sequence cap "
            f"is {DIAGNOSTIC_SEQUENCE_CAP} tokens"
        )
    record_bytes = grouped_causal_diagnostic_record_bytes(q, k, v)
    if record_bytes > DIAGNOSTIC_RECORD_BYTE_BUDGET:
        raise ValueError(
            "grouped causal diagnostic record budget "
            f"is {DIAGNOSTIC_RECORD_BYTE_BUDGET} bytes, but geometry requires "
            f"{record_bytes} bytes"
        )


def default_grouped_causal_scan_config(
    *,
    dtype: torch.dtype,
    head_dimension: int,
    value_dimension: int,
) -> GroupedCausalScanConfig:
    """Return the deterministic baseline schedule for supported causal scans.

    The configuration is deliberately conservative.  The future Triton
    token/pair-wave launchers consume these validated values; the current
    correctness-first implementation records its scalar token schedule
    separately in diagnostics.
    """
    if dtype not in {torch.float16, torch.bfloat16}:
        raise TypeError("grouped causal scan scheduling supports float16 and bfloat16")
    _require_positive(
        head_dimension=head_dimension,
        value_dimension=value_dimension,
    )
    if head_dimension > 64 or value_dimension > 64:
        raise ValueError(
            "grouped causal scan scheduling supports dimensions through 64"
        )
    return GroupedCausalScanConfig(
        token_block=16,
        pair_block=64,
        value_block=16,
        groups_per_wave=4,
    )


def grouped_causal_state_bytes(
    *,
    batch_size: int,
    key_value_heads: int,
    head_dimension: int,
    value_dimension: int,
) -> int:
    """Return FP32 bytes for one KV-owned H or A state set.

    The state deliberately omits sequence length: scans retain only H0, H1,
    and packed-symmetric H2 at their current causal position.
    """
    _require_positive(
        batch_size=batch_size,
        key_value_heads=key_value_heads,
        head_dimension=head_dimension,
        value_dimension=value_dimension,
    )
    pair_count = head_dimension * (head_dimension + 1) // 2
    return (
        _FP32_BYTES
        * batch_size
        * key_value_heads
        * (1 + head_dimension + pair_count)
        * (value_dimension + 1)
    )


def canonical_group_indices(
    dim_groups: torch.Tensor,
    *,
    query_heads: int,
    head_dimension: int,
    gmax: int,
) -> torch.Tensor:
    """Return small ``[Hq,D]`` long indices without creating token state."""
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


__all__ = (
    "DIAGNOSTIC_MACRO_TOKEN_BLOCK",
    "DIAGNOSTIC_RECORD_BYTE_BUDGET",
    "DIAGNOSTIC_SEQUENCE_CAP",
    "PHYSICAL_PATH_IDENTIFIER",
    "KernelGeometry",
    "KernelPlan",
    "GroupedCausalScanConfig",
    "GroupedCausalBackwardDiagnosticWitness",
    "GroupedCausalForwardDiagnosticWitness",
    "GroupedCausalForwardTokenWitness",
    "GroupedCausalStateSnapshot",
    "GroupedCausalSuffixStateSnapshot",
    "GroupedCausalWorkspaceAccounting",
    "WorkspaceAllocation",
    "WorkspacePlan",
    "build_kernel_plan",
    "canonical_packed_pair_metadata",
    "canonical_group_indices",
    "default_grouped_causal_scan_config",
    "grouped_causal_state_bytes",
    "grouped_causal_diagnostic_record_bytes",
    "planned_grouped_causal_scan_config",
    "require_cpu_diagnostic_geometry",
)
