"""Public metadata-only admission for one exact grouped-quadratic execution."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Literal, Mapping

import torch

from .triton.grouped_quadratic_capabilities import (
    GroupedQuadraticCapabilityKey,
    ReviewedCapabilityEvidence,
    reviewed_evidence_for_capability_key,
)
from .triton.grouped_quadratic_causal_backward import (
    _backward_stage_source_hash,
    _normalization_stage_source_hash,
    _requires_prefix_vjp,
    build_normalization_kernel_plan_for_geometry,
)
from .triton.grouped_quadratic_causal_common import (
    PHYSICAL_PATH_IDENTIFIER,
    KernelGeometry,
    build_kernel_plan,
    collect_cuda_device_properties,
)
from .triton.grouped_quadratic_causal_forward_kernels import _source_hash

_SCHEMA_VERSION = "hd_mgq_grouped_execution_admission_v1"
_EXECUTION_MODES = frozenset(
    {
        "forward",
        "backward_normalization",
        "backward_prefix",
        "backward_suffix",
    }
)
_RUNTIME_FIELDS = (
    "sm_count",
    "warp_size",
    "max_threads_per_block",
    "max_threads_per_sm",
    "shared_memory_per_block_optin",
    "shared_memory_per_sm",
    "registers_per_block",
    "registers_per_sm",
    "l2_bytes",
    "memory_bus_width_bits",
    "memory_clock_rate",
)


class GroupedExecutionAdmissionError(RuntimeError):
    """A metadata-only grouped request has no exact public admission."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _frozen_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class GroupedExecutionAdmissionStage:
    """Static plan inputs for one semantic execution stage.

    Device and toolchain fields are deliberately absent: The operator package discovers those
    itself from the selected runtime.  The remaining fields are the planner
    inputs that must match a later real grouped execution exactly.
    """

    name: str
    execution_modes: tuple[str, ...]
    dtype: str
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
    requested_gradient_mask: tuple[bool, bool, bool, bool, bool, bool]
    workspace_budget_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("admission stage name must be a non-empty string")
        modes = tuple(self.execution_modes)
        if (
            not modes
            or len(set(modes)) != len(modes)
            or any(mode not in _EXECUTION_MODES for mode in modes)
        ):
            raise ValueError(
                "admission stage execution modes must be unique known physical modes"
            )
        if self.dtype not in {"float16", "bfloat16"}:
            raise ValueError("admission stage dtype must be float16 or bfloat16")
        for field_name in (
            "batch_size",
            "query_heads",
            "key_value_heads",
            "sequence_length",
            "head_dimension",
            "value_dimension",
            "gmax",
            "workspace_budget_bytes",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(
                    f"admission stage {field_name} must be a positive integer"
                )
        if self.query_heads % self.key_value_heads:
            raise ValueError(
                "admission stage query_heads must divide by key_value_heads"
            )
        if not isinstance(self.causal, bool) or not self.causal:
            raise ValueError(
                "admission stage must use the causal grouped physical path"
            )
        if self.group_layout not in {"shared", "per_head"}:
            raise ValueError("admission stage group_layout must be shared or per_head")
        if self.coefficient_layout not in {"shared", "per_head"}:
            raise ValueError(
                "admission stage coefficient_layout must be shared or per_head"
            )
        if self.stride_class not in {"contiguous", "constant_head_strided"}:
            raise ValueError("admission stage has an unsupported stride_class")
        mask = tuple(self.requested_gradient_mask)
        if len(mask) != 6 or any(not isinstance(value, bool) for value in mask):
            raise TypeError(
                "admission stage requested_gradient_mask must contain six booleans"
            )
        _validate_requested_modes(modes, mask)
        object.__setattr__(self, "execution_modes", modes)
        object.__setattr__(self, "requested_gradient_mask", mask)

    def to_mapping(self) -> dict[str, object]:
        return {
            "name": self.name,
            "execution_modes": list(self.execution_modes),
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
        }


def _validate_requested_modes(
    execution_modes: tuple[str, ...],
    requested_mask: tuple[bool, bool, bool, bool, bool, bool],
) -> None:
    prefix_required = _requires_prefix_vjp(requested_mask)
    suffix_required = requested_mask[1] or requested_mask[2]
    normalization_required = any(requested_mask)
    requested = set(execution_modes)
    if "forward" not in requested:
        raise ValueError("admission stage must include the forward physical mode")
    expected_backward = {
        mode
        for mode, required in (
            ("backward_normalization", normalization_required),
            ("backward_prefix", prefix_required),
            ("backward_suffix", suffix_required),
        )
        if required
    }
    supplied_backward = requested - {"forward"}
    if supplied_backward != expected_backward:
        raise ValueError(
            "admission stage physical modes do not match requested gradients"
        )


@dataclass(frozen=True)
class GroupedExecutionAdmissionRequest:
    """Canonical public request for metadata-only exact-record lookup."""

    device_index: int
    stages: tuple[GroupedExecutionAdmissionStage, ...]
    schema_version: Literal["hd_mgq_grouped_execution_admission_v1"] = _SCHEMA_VERSION
    physical_path: Literal["hadamard_h012_packed_diag_v1"] = PHYSICAL_PATH_IDENTIFIER

    def __post_init__(self) -> None:
        if (
            not isinstance(self.device_index, int)
            or isinstance(self.device_index, bool)
            or self.device_index < 0
        ):
            raise ValueError("admission device_index must be a non-negative integer")
        stages = tuple(self.stages)
        if not stages or any(
            type(stage) is not GroupedExecutionAdmissionStage for stage in stages
        ):
            raise TypeError(
                "admission stages must be non-empty GroupedExecutionAdmissionStage values"
            )
        names = tuple(stage.name for stage in stages)
        if len(set(names)) != len(names):
            raise ValueError("admission stage names must be unique")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("unsupported grouped execution admission schema")
        if self.physical_path != PHYSICAL_PATH_IDENTIFIER:
            raise ValueError(
                "admission physical path must be hadamard_h012_packed_diag_v1"
            )
        object.__setattr__(self, "stages", stages)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "physical_path": self.physical_path,
            "device_index": self.device_index,
            "stages": [stage.to_mapping() for stage in self.stages],
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.to_mapping())

    @property
    def request_sha256(self) -> str:
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GroupedExecutionAdmission:
    """Immutable plan resolution and reviewed-evidence report for one request."""

    request_sha256: str
    runtime_descriptor: Mapping[str, object]
    record_identities: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        if not _is_hex_digest(self.request_sha256):
            raise ValueError("admission request_sha256 must be lowercase SHA-256")
        if (
            not isinstance(self.runtime_descriptor, Mapping)
            or not self.runtime_descriptor
        ):
            raise TypeError("admission runtime_descriptor must be a non-empty mapping")
        identities = tuple(self.record_identities)
        if not identities or any(
            not isinstance(identity, Mapping) for identity in identities
        ):
            raise TypeError("admission record_identities must be non-empty mappings")
        object.__setattr__(
            self, "runtime_descriptor", _frozen_mapping(self.runtime_descriptor)
        )
        object.__setattr__(
            self,
            "record_identities",
            tuple(_frozen_mapping(identity) for identity in identities),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "request_sha256": self.request_sha256,
            "runtime_descriptor": dict(self.runtime_descriptor),
            "record_identities": [
                dict(identity) for identity in self.record_identities
            ],
        }


def _is_hex_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _inspect_runtime_descriptor(device_index: int) -> dict[str, object]:
    """Read only immutable device/runtime properties; do not allocate or launch."""
    if not torch.cuda.is_available():
        raise GroupedExecutionAdmissionError(
            "CUDA runtime is unavailable for metadata-only admission"
        )
    if device_index >= torch.cuda.device_count():
        raise GroupedExecutionAdmissionError(
            "admission device_index is outside the available CUDA devices"
        )
    try:
        from .triton.grouped_quadratic_causal_forward_kernels import triton

        if triton is None:
            raise RuntimeError("Triton is unavailable")
        properties = torch.cuda.get_device_properties(device_index)
        hardware = collect_cuda_device_properties(
            properties,
            device_index=device_index,
            triton_module=triton,
        )
    except Exception as exc:
        raise GroupedExecutionAdmissionError(
            "unable to collect complete CUDA runtime facts for metadata-only admission"
        ) from exc
    return {
        "compute_capability": (properties.major, properties.minor),
        **hardware,
        "torch_version": torch.__version__,
        "triton_version": str(getattr(triton, "__version__", "unknown")),
    }


def _runtime_geometry(
    stage: GroupedExecutionAdmissionStage,
    runtime_descriptor: Mapping[str, object],
    *,
    requested_gradient_mask: tuple[bool, bool, bool, bool, bool, bool],
    source_hash: str,
    launch_stage: str = "composite",
) -> KernelGeometry:
    compute_capability = runtime_descriptor.get("compute_capability")
    if (
        not isinstance(compute_capability, tuple)
        or len(compute_capability) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in compute_capability
        )
    ):
        raise GroupedExecutionAdmissionError(
            "runtime descriptor has an invalid compute capability"
        )
    values: dict[str, object] = {}
    for field_name in _RUNTIME_FIELDS:
        value = runtime_descriptor.get(field_name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise GroupedExecutionAdmissionError(
                f"runtime descriptor has no positive {field_name}"
            )
        values[field_name] = value
    torch_version = runtime_descriptor.get("torch_version")
    triton_version = runtime_descriptor.get("triton_version")
    if (
        not isinstance(torch_version, str)
        or not torch_version
        or not isinstance(triton_version, str)
        or not triton_version
    ):
        raise GroupedExecutionAdmissionError(
            "runtime descriptor is missing a toolchain version"
        )
    return KernelGeometry(
        compute_capability=compute_capability,
        **values,  # type: ignore[arg-type]
        torch_version=torch_version,
        triton_version=triton_version,
        source_hash=source_hash,
        dtype=stage.dtype,
        batch_size=stage.batch_size,
        query_heads=stage.query_heads,
        key_value_heads=stage.key_value_heads,
        sequence_length=stage.sequence_length,
        head_dimension=stage.head_dimension,
        value_dimension=stage.value_dimension,
        gmax=stage.gmax,
        causal=stage.causal,
        group_layout=stage.group_layout,
        coefficient_layout=stage.coefficient_layout,
        stride_class=stage.stride_class,
        requested_gradient_mask=requested_gradient_mask,
        workspace_budget_bytes=stage.workspace_budget_bytes,
        launch_stage=launch_stage,
    )


def _resolve_stage_capability_keys(
    stage: GroupedExecutionAdmissionStage,
    runtime_descriptor: Mapping[str, object],
) -> tuple[GroupedQuadraticCapabilityKey, ...]:
    """Resolve exactly the forward/VJP plans that real execution would require."""
    full_mask = stage.requested_gradient_mask
    prefix_mask = (full_mask[0], False, False, *full_mask[3:])
    suffix_mask = (False, full_mask[1], full_mask[2], False, False, False)
    resolved: list[GroupedQuadraticCapabilityKey] = []
    for mode in stage.execution_modes:
        if mode == "forward":
            plan = build_kernel_plan(
                _runtime_geometry(
                    stage,
                    runtime_descriptor,
                    requested_gradient_mask=full_mask,
                    source_hash=_source_hash(),
                )
            )
        elif mode == "backward_normalization":
            plan = build_normalization_kernel_plan_for_geometry(
                _runtime_geometry(
                    stage,
                    runtime_descriptor,
                    requested_gradient_mask=full_mask,
                    source_hash=_normalization_stage_source_hash(),
                    launch_stage="normalization_vjp",
                )
            )
        elif mode == "backward_prefix":
            plan = build_kernel_plan(
                _runtime_geometry(
                    stage,
                    runtime_descriptor,
                    requested_gradient_mask=prefix_mask,
                    source_hash=_backward_stage_source_hash("prefix"),
                )
            )
        else:
            assert mode == "backward_suffix"
            plan = build_kernel_plan(
                _runtime_geometry(
                    stage,
                    runtime_descriptor,
                    requested_gradient_mask=suffix_mask,
                    source_hash=_backward_stage_source_hash("suffix"),
                )
            )
        resolved.append(
            GroupedQuadraticCapabilityKey(
                plan=plan,
                execution_mode=mode,  # type: ignore[arg-type]
                requested_gradient_mask=plan.geometry.requested_gradient_mask,
            )
        )
    return tuple(resolved)


def _record_identity(
    *,
    stage_name: str,
    key: GroupedQuadraticCapabilityKey,
    evidence: ReviewedCapabilityEvidence,
) -> dict[str, object]:
    key_json = key.to_json().encode("utf-8")
    identity: dict[str, object] = {
        "stage_name": stage_name,
        "execution_mode": key.execution_mode,
        "capability_key_sha256": sha256(key_json).hexdigest(),
        "evidence_status": evidence.status,
    }
    if evidence.status == "current":
        assert evidence.record is not None
        digest = evidence.record.get("record_payload_sha256")
        if not _is_hex_digest(digest):
            raise GroupedExecutionAdmissionError(
                "reviewed record has no valid payload digest"
            )
        identity["record_payload_sha256"] = digest
    diagnostic_kinds = tuple(dict.fromkeys(item.kind for item in evidence.diagnostics))
    if diagnostic_kinds:
        identity["evidence_diagnostics"] = diagnostic_kinds
    return identity


def resolve_grouped_execution_admission(
    request: GroupedExecutionAdmissionRequest,
) -> GroupedExecutionAdmission:
    """Resolve one actual-device request without creating tensors or launching kernels."""
    if type(request) is not GroupedExecutionAdmissionRequest:
        raise TypeError(
            "public grouped admission requires a GroupedExecutionAdmissionRequest"
        )
    runtime_descriptor = _inspect_runtime_descriptor(request.device_index)
    identities: list[Mapping[str, object]] = []
    for stage in request.stages:
        for key in _resolve_stage_capability_keys(stage, runtime_descriptor):
            evidence = reviewed_evidence_for_capability_key(key)
            identities.append(
                _record_identity(stage_name=stage.name, key=key, evidence=evidence)
            )
    return GroupedExecutionAdmission(
        request_sha256=request.request_sha256,
        runtime_descriptor=runtime_descriptor,
        record_identities=tuple(identities),
    )


__all__ = (
    "GroupedExecutionAdmission",
    "GroupedExecutionAdmissionError",
    "GroupedExecutionAdmissionRequest",
    "GroupedExecutionAdmissionStage",
    "resolve_grouped_execution_admission",
)
