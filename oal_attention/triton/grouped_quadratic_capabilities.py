"""Non-blocking reviewed test evidence for grouped quadratic Triton.

Execution is selected from the live device, dtype, geometry, planner, and
workspace constraints.  Reviewed records report whether an exact planned path
has current test evidence; they never serve as runtime licenses.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from hmac import compare_digest
import json
from pathlib import Path
from typing import Any, Literal, TypeAlias
from types import MappingProxyType

import torch

from .grouped_quadratic_causal_common import KernelPlan, PHYSICAL_PATH_IDENTIFIER

GroupedQuadraticCapability: TypeAlias = tuple[
    torch.dtype,
    int,
    int,
    int,
    int,
    int,
    int,
    bool,
    int,
]
# Historical field order: dtype, B, Hq, Hkv, N, D, DV, causal, Gmax.


# These tuples documented the retired source-review boundary.  They must not
# authorize a new physical path merely because its tensor geometry matches.
LEGACY_GROUPED_QUADRATIC_CAPABILITIES: frozenset[GroupedQuadraticCapability] = (
    frozenset(
        {
            (torch.float16, 1, 14, 2, 4096, 64, 64, True, 2),
            (torch.bfloat16, 1, 14, 2, 4096, 64, 64, True, 2),
            (torch.bfloat16, 1, 14, 2, 8192, 64, 64, True, 2),
        }
    )
)

# Kept as an empty compatibility value while the planner has no exact plan key.
REVIEWED_GROUPED_QUADRATIC_CAPABILITIES: frozenset[GroupedQuadraticCapability] = (
    frozenset()
)

_REVIEWED_RECORDS_PATH = Path(__file__).with_name(
    "grouped_quadratic_reviewed_records.json"
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# This is deliberately a code-only manifest.  The append-only records data is
# validated entry-by-entry and is never folded into a self-referential digest.
GROUPED_QUADRATIC_CAPABILITY_CODE_MANIFEST_FILES: tuple[str, ...] = (
    "oal_attention/grouped_quadratic.py",
    "oal_attention/grouped_quadratic_admission.py",
    "oal_attention/triton/grouped_quadratic.py",
    "oal_attention/triton/grouped_quadratic_backward.py",
    "oal_attention/triton/grouped_quadratic_capabilities.py",
    "oal_attention/triton/grouped_quadratic_causal_common.py",
    "oal_attention/triton/grouped_quadratic_causal_forward.py",
    "oal_attention/triton/grouped_quadratic_causal_forward_kernels.py",
    "oal_attention/triton/grouped_quadratic_causal_backward.py",
    "oal_attention/triton/grouped_quadratic_causal_backward_kernels.py",
    "oal_attention/triton/grouped_quadratic_causal_prefix_kernels.py",
)


def _canonical_json_value(value: object) -> object:
    """Convert mutable or frozen JSON containers to canonical JSON values."""
    if isinstance(value, Mapping):
        return {
            key: _canonical_json_value(child_value)
            for key, child_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(child_value) for child_value in value]
    return value


def _freeze_json_value(value: object) -> object:
    """Recursively freeze JSON containers after their raw payload is validated."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json_value(child_value) for key, child_value in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json_value(child_value) for child_value in value)
    return value


def _freeze_reviewed_record(record: Mapping[str, object]) -> Mapping[str, object]:
    frozen = _freeze_json_value(record)
    assert isinstance(frozen, Mapping)
    return frozen


def grouped_quadratic_capability_code_manifest_sha256() -> str:
    """Hash executable admission code, excluding mutable reviewed-record data."""
    digest = sha256()
    for relative_path in GROUPED_QUADRATIC_CAPABILITY_CODE_MANIFEST_FILES:
        contents_digest = sha256(
            (_REPOSITORY_ROOT / relative_path).read_bytes()
        ).digest()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(contents_digest)
    return digest.hexdigest()


def canonical_record_payload(record: Mapping[str, object]) -> bytes:
    """Return canonical UTF-8 record bytes with only its digest omitted."""
    payload = {
        key: _canonical_json_value(value)
        for key, value in record.items()
        if key != "record_payload_sha256"
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def reviewed_record_is_valid(record: Mapping[str, object]) -> bool:
    """Validate one complete current-source reviewed-evidence record."""
    supplied_digest = record.get("record_payload_sha256")
    supplied_manifest = record.get("code_manifest_sha256")
    if (
        not _is_sha256(supplied_digest)
        or not _is_sha256(supplied_manifest)
        or not _record_schema_is_complete(record)
    ):
        return False
    if not compare_digest(
        supplied_manifest,
        grouped_quadratic_capability_code_manifest_sha256(),
    ):
        return False
    expected_digest = sha256(canonical_record_payload(record)).hexdigest()
    return compare_digest(supplied_digest, expected_digest)


def reviewed_records_are_valid(records: Sequence[Mapping[str, object]]) -> bool:
    """Check each record independently so appending cannot change prior hashes."""
    return all(reviewed_record_is_valid(record) for record in records)


@dataclass(frozen=True)
class ReviewedRecordDiagnostic:
    """One non-fatal problem discovered while reading reviewed evidence."""

    kind: str
    message: str
    index: int | None = None


@dataclass(frozen=True)
class ReviewedRecordStatus:
    """Source-currency state of one immutable record payload."""

    index: int
    status: Literal["current", "stale", "invalid"]
    record: Mapping[str, object]


@dataclass(frozen=True)
class ReviewedRecordStore:
    """Import-safe reviewed evidence plus deterministic diagnostics."""

    records: tuple[Mapping[str, object], ...]
    record_statuses: tuple[ReviewedRecordStatus, ...]
    diagnostics: tuple[ReviewedRecordDiagnostic, ...]

    @property
    def current_records(self) -> tuple[Mapping[str, object], ...]:
        return tuple(
            item.record for item in self.record_statuses if item.status == "current"
        )


@dataclass(frozen=True)
class ReviewedCapabilityEvidence:
    """Non-blocking evidence result for one exact planned capability key."""

    status: Literal["current", "stale", "unreviewed"]
    record: Mapping[str, object] | None
    diagnostics: tuple[ReviewedRecordDiagnostic, ...]


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _is_nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _is_nonempty_str(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _is_six_bool_mask(value: object) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 6
        and all(type(item) is bool for item in value)
    )


_REVIEW_SCHEMA_VERSION = "hd_mgq_independent_candidate_review_v1"
_CAPABILITY_KEY_SCHEMA_VERSION = "grouped_quadratic_capability_key_v1"
_CAPABILITY_KEY_FIELDS = frozenset(
    {
        "execution_mode",
        "geometry",
        "gqa_ratio",
        "physical_path",
        "plan_id",
        "planner",
        "requested_gradient_mask",
        "schema_version",
        "tail_class",
        "workspace_budget_class",
    }
)
_GEOMETRY_FIELDS = frozenset(
    {
        "batch_size",
        "causal",
        "coefficient_layout",
        "compute_capability",
        "dtype",
        "gmax",
        "group_layout",
        "head_dimension",
        "key_value_heads",
        "l2_bytes",
        "launch_stage",
        "max_threads_per_block",
        "max_threads_per_sm",
        "memory_bus_width_bits",
        "memory_clock_rate",
        "query_heads",
        "registers_per_block",
        "registers_per_sm",
        "requested_gradient_mask",
        "sequence_length",
        "shared_memory_per_block_optin",
        "shared_memory_per_sm",
        "sm_count",
        "source_hash",
        "stride_class",
        "torch_version",
        "triton_version",
        "value_dimension",
        "warp_size",
        "workspace_budget_bytes",
    }
)
_PLANNER_FIELDS = frozenset(
    {
        "canonical_pair_order",
        "runtime_axes",
        "source_hash",
        "specialization_axes",
        "version",
    }
)
_RUNTIME_AXES_FIELDS = frozenset(
    {
        "batch_size",
        "gqa_ratio",
        "key_value_heads",
        "pair_count",
        "query_heads",
        "sequence_length",
    }
)
_SPECIALIZATION_AXES_FIELDS = frozenset(
    {
        "causal",
        "dtype",
        "gmax",
        "groups_per_wave",
        "head_dimension",
        "launch_stage",
        "macro_token_block",
        "macro_token_block_selection",
        "pair_block",
        "reduction_replicas",
        "value_block",
        "value_dimension",
    }
)
_INDEPENDENT_REVIEW_FIELDS = frozenset(
    {
        "candidate_file_sha256",
        "candidate_filename",
        "candidate_record_payload_sha256",
        "capture_manifest_sha256",
        "config_sha256",
        "formal_asset_sha256",
        "physical_stage",
        "semantic_request",
    }
)


def _record_schema_is_complete(record: Mapping[str, object]) -> bool:
    capability_key = record.get("capability_key")
    independent_review = record.get("independent_review")
    if (
        record.get("review_schema_version") != _REVIEW_SCHEMA_VERSION
        or not isinstance(capability_key, Mapping)
        or not isinstance(independent_review, Mapping)
        or not _CAPABILITY_KEY_FIELDS.issubset(capability_key)
        or not _INDEPENDENT_REVIEW_FIELDS.issubset(independent_review)
    ):
        return False
    geometry = capability_key.get("geometry")
    planner = capability_key.get("planner")
    requested_mask = capability_key.get("requested_gradient_mask")
    execution_mode = capability_key.get("execution_mode")
    if (
        capability_key.get("schema_version") != _CAPABILITY_KEY_SCHEMA_VERSION
        or not _is_sha256(capability_key.get("plan_id"))
        or execution_mode
        not in {
            "forward",
            "backward",
            "backward_normalization",
            "backward_prefix",
            "backward_suffix",
        }
        or not _is_positive_int(capability_key.get("gqa_ratio"))
        or capability_key.get("physical_path") != PHYSICAL_PATH_IDENTIFIER
        or not isinstance(geometry, Mapping)
        or not _GEOMETRY_FIELDS.issubset(geometry)
        or not isinstance(planner, Mapping)
        or not _PLANNER_FIELDS.issubset(planner)
        or not _is_six_bool_mask(requested_mask)
    ):
        return False
    geometry_positive_int_fields = _GEOMETRY_FIELDS - {
        "causal",
        "coefficient_layout",
        "compute_capability",
        "dtype",
        "group_layout",
        "launch_stage",
        "requested_gradient_mask",
        "source_hash",
        "stride_class",
        "torch_version",
        "triton_version",
    }
    geometry_string_fields = {
        "coefficient_layout",
        "dtype",
        "group_layout",
        "launch_stage",
        "stride_class",
        "torch_version",
        "triton_version",
    }
    compute_capability = geometry.get("compute_capability")
    geometry_mask = geometry.get("requested_gradient_mask")
    if (
        type(geometry.get("causal")) is not bool
        or not all(
            _is_positive_int(geometry.get(field))
            for field in geometry_positive_int_fields
        )
        or not all(
            _is_nonempty_str(geometry.get(field)) for field in geometry_string_fields
        )
        or geometry.get("dtype") not in {"float16", "bfloat16"}
        or geometry.get("group_layout") not in {"shared", "per_head"}
        or geometry.get("coefficient_layout") not in {"shared", "per_head"}
        or geometry.get("stride_class") != "contiguous"
        or geometry.get("launch_stage") not in {"composite", "normalization_vjp"}
        or not _is_sha256(geometry.get("source_hash"))
        or not isinstance(compute_capability, (list, tuple))
        or len(compute_capability) != 2
        or not _is_positive_int(compute_capability[0])
        or not _is_nonnegative_int(compute_capability[1])
        or not _is_six_bool_mask(geometry_mask)
        or tuple(geometry_mask) != tuple(requested_mask)
        or geometry.get("query_heads")
        != geometry.get("key_value_heads") * capability_key.get("gqa_ratio")
    ):
        return False
    canonical_pair_order = planner.get("canonical_pair_order")
    runtime_axes = planner.get("runtime_axes")
    specialization_axes = planner.get("specialization_axes")
    if (
        not _is_sha256(planner.get("source_hash"))
        or not _is_nonempty_str(planner.get("version"))
        or not isinstance(canonical_pair_order, Mapping)
        or canonical_pair_order.get("kind") != "lower_triangular_row_major"
        or canonical_pair_order.get("formula") != "p(r,s)=r*(r+1)//2+s"
        or not isinstance(runtime_axes, Mapping)
        or not _RUNTIME_AXES_FIELDS.issubset(runtime_axes)
        or not all(
            _is_positive_int(runtime_axes.get(field)) for field in _RUNTIME_AXES_FIELDS
        )
        or not isinstance(specialization_axes, Mapping)
        or not _SPECIALIZATION_AXES_FIELDS.issubset(specialization_axes)
    ):
        return False
    runtime_geometry_pairs = {
        "batch_size": "batch_size",
        "gqa_ratio": "gqa_ratio",
        "key_value_heads": "key_value_heads",
        "query_heads": "query_heads",
        "sequence_length": "sequence_length",
    }
    if any(
        runtime_axes.get(runtime_field)
        != (
            capability_key.get("gqa_ratio")
            if geometry_field == "gqa_ratio"
            else geometry.get(geometry_field)
        )
        for runtime_field, geometry_field in runtime_geometry_pairs.items()
    ) or runtime_axes.get("pair_count") != (
        geometry.get("head_dimension") * (geometry.get("head_dimension") + 1) // 2
    ):
        return False
    specialization_int_fields = _SPECIALIZATION_AXES_FIELDS - {
        "causal",
        "dtype",
        "launch_stage",
        "macro_token_block_selection",
    }
    if (
        type(specialization_axes.get("causal")) is not bool
        or not all(
            _is_positive_int(specialization_axes.get(field))
            for field in specialization_int_fields
        )
        or not _is_nonempty_str(specialization_axes.get("dtype"))
        or not _is_nonempty_str(specialization_axes.get("launch_stage"))
        or not _is_nonempty_str(specialization_axes.get("macro_token_block_selection"))
        or specialization_axes.get("causal") != geometry.get("causal")
        or specialization_axes.get("dtype") != geometry.get("dtype")
        or specialization_axes.get("gmax") != geometry.get("gmax")
        or specialization_axes.get("head_dimension") != geometry.get("head_dimension")
        or specialization_axes.get("launch_stage") != geometry.get("launch_stage")
        or specialization_axes.get("value_dimension") != geometry.get("value_dimension")
    ):
        return False
    if (
        capability_key.get("tail_class")
        != f"tail:{geometry.get('sequence_length') % specialization_axes.get('macro_token_block')}"
        or capability_key.get("workspace_budget_class")
        != f"exact_bytes:{geometry.get('workspace_budget_bytes')}"
    ):
        return False
    review_sha_fields = _INDEPENDENT_REVIEW_FIELDS - {
        "candidate_filename",
        "physical_stage",
        "semantic_request",
    }
    physical_stage_by_mode = {
        "forward": "forward",
        "backward": "backward",
        "backward_normalization": "normalization",
        "backward_prefix": "prefix",
        "backward_suffix": "suffix",
    }
    return (
        all(_is_sha256(independent_review.get(field)) for field in review_sha_fields)
        and isinstance(independent_review.get("candidate_filename"), str)
        and bool(independent_review.get("candidate_filename"))
        and isinstance(independent_review.get("physical_stage"), str)
        and independent_review.get("physical_stage")
        == physical_stage_by_mode[execution_mode]
        and isinstance(independent_review.get("semantic_request"), str)
        and bool(independent_review.get("semantic_request"))
    )


def _record_payload_is_intact(record: Mapping[str, object]) -> bool:
    supplied_digest = record.get("record_payload_sha256")
    supplied_manifest = record.get("code_manifest_sha256")
    capability_key = record.get("capability_key")
    if (
        not _is_sha256(supplied_digest)
        or not _is_sha256(supplied_manifest)
        or not isinstance(capability_key, Mapping)
        or not _record_schema_is_complete(record)
    ):
        return False
    expected_digest = sha256(canonical_record_payload(record)).hexdigest()
    return compare_digest(supplied_digest, expected_digest)


def _reviewed_record_store_from_values(
    raw_values: Sequence[object],
    *,
    initial_diagnostics: Sequence[ReviewedRecordDiagnostic] = (),
) -> ReviewedRecordStore:
    records: list[Mapping[str, object]] = []
    statuses: list[ReviewedRecordStatus] = []
    diagnostics = list(initial_diagnostics)
    current_manifest = grouped_quadratic_capability_code_manifest_sha256()
    for index, value in enumerate(raw_values):
        if not isinstance(value, Mapping):
            diagnostics.append(
                ReviewedRecordDiagnostic(
                    "invalid_record", "reviewed record entry is not an object", index
                )
            )
            continue
        frozen = _freeze_reviewed_record(value)
        records.append(frozen)
        if not _record_payload_is_intact(frozen):
            status: Literal["current", "stale", "invalid"] = "invalid"
            diagnostics.append(
                ReviewedRecordDiagnostic(
                    "invalid_record",
                    "reviewed record payload, manifest, or capability key is invalid",
                    index,
                )
            )
        elif compare_digest(str(frozen["code_manifest_sha256"]), current_manifest):
            status = "current"
        else:
            status = "stale"
            diagnostics.append(
                ReviewedRecordDiagnostic(
                    "stale_record",
                    "reviewed record belongs to an older source manifest",
                    index,
                )
            )
        statuses.append(ReviewedRecordStatus(index, status, frozen))
    return ReviewedRecordStore(tuple(records), tuple(statuses), tuple(diagnostics))


def _load_reviewed_record_store(
    path: Path = _REVIEWED_RECORDS_PATH,
) -> ReviewedRecordStore:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ReviewedRecordStore(
            (),
            (),
            (
                ReviewedRecordDiagnostic(
                    "missing_file", "reviewed record file is missing"
                ),
            ),
        )
    except UnicodeError as exc:
        return ReviewedRecordStore(
            (),
            (),
            (
                ReviewedRecordDiagnostic(
                    "invalid_json", f"reviewed record JSON is invalid: {exc}"
                ),
            ),
        )
    except OSError as exc:
        return ReviewedRecordStore(
            (),
            (),
            (
                ReviewedRecordDiagnostic(
                    "unreadable_file", f"reviewed record file is unreadable: {exc}"
                ),
            ),
        )
    try:
        raw_records = json.loads(raw_text)
    except (json.JSONDecodeError, UnicodeError) as exc:
        return ReviewedRecordStore(
            (),
            (),
            (
                ReviewedRecordDiagnostic(
                    "invalid_json", f"reviewed record JSON is invalid: {exc}"
                ),
            ),
        )
    if not isinstance(raw_records, list):
        return ReviewedRecordStore(
            (),
            (),
            (
                ReviewedRecordDiagnostic(
                    "non_array_root", "reviewed record root is not an array"
                ),
            ),
        )
    return _reviewed_record_store_from_values(raw_records)


def _load_reviewed_records() -> tuple[Mapping[str, object], ...]:
    """Compatibility wrapper returning immutable records without import failure."""
    return _load_reviewed_record_store(_REVIEWED_RECORDS_PATH).records


# Legacy evidence readers below remain available to historical reporting tools.
# Training imports never read a source manifest or reviewed-record registry.
REVIEWED_GROUPED_QUADRATIC_RECORD_STORE = ReviewedRecordStore((), (), ())
REVIEWED_GROUPED_QUADRATIC_RECORDS = REVIEWED_GROUPED_QUADRATIC_RECORD_STORE.records
REVIEWED_GROUPED_QUADRATIC_RECORD_STATUSES = (
    REVIEWED_GROUPED_QUADRATIC_RECORD_STORE.record_statuses
)
REVIEWED_GROUPED_QUADRATIC_RECORD_DIAGNOSTICS = (
    REVIEWED_GROUPED_QUADRATIC_RECORD_STORE.diagnostics
)


@dataclass(frozen=True)
class GroupedQuadraticCapabilityKey:
    """Immutable, JSON-safe identity for one exact planned launch stage.

    A reviewed record describes evidence for this complete plan payload, one
    stage (forward or backward), and the exact Q/K/V/A/B/C gradient request.
    The plan retains all scheduling axes so evidence cannot silently migrate
    to a different tail, workspace, or CUDA resource/toolchain tuple.
    """

    plan: KernelPlan
    execution_mode: Literal[
        "forward",
        "backward",
        "backward_normalization",
        "backward_prefix",
        "backward_suffix",
    ]
    requested_gradient_mask: tuple[bool, bool, bool, bool, bool, bool]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, KernelPlan):
            raise TypeError("capability keys require a KernelPlan")
        if self.execution_mode not in {
            "forward",
            "backward",
            "backward_normalization",
            "backward_prefix",
            "backward_suffix",
        }:
            raise ValueError(
                "capability execution_mode must be forward, backward, backward_normalization, backward_prefix, or backward_suffix"
            )
        mask = tuple(self.requested_gradient_mask)
        if len(mask) != 6 or any(not isinstance(value, bool) for value in mask):
            raise TypeError(
                "capability gradient mask must contain six booleans for Q/K/V/A/B/C"
            )
        if mask != self.plan.geometry.requested_gradient_mask:
            raise ValueError(
                "capability gradient mask must exactly match the planned geometry"
            )
        object.__setattr__(self, "requested_gradient_mask", mask)

    @property
    def gqa_ratio(self) -> int:
        return self.plan.geometry.Hq // self.plan.geometry.Hkv

    @property
    def plan_id(self) -> str:
        """Expose the immutable planner identity without a mutable copy."""
        return self.plan.plan_id

    @property
    def tail_class(self) -> str:
        macro_token_block = self.plan.specialization_axes["macro_token_block"]
        if not isinstance(macro_token_block, int) or macro_token_block <= 0:
            raise ValueError("capability plan has no positive macro_token_block")
        return f"tail:{self.plan.geometry.N % macro_token_block}"

    @property
    def workspace_budget_class(self) -> str:
        budget = self.plan.geometry.workspace_budget_bytes
        if not isinstance(budget, int) or budget <= 0:
            raise ValueError("capability plan must carry an exact workspace budget")
        # This intentionally is not a bucket: one exact planned budget equals
        # one evidence record, so no adjacent free-memory tuple can inherit it.
        return f"exact_bytes:{budget}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "grouped_quadratic_capability_key_v1",
            "physical_path": self.plan.physical_path,
            "plan_id": self.plan.plan_id,
            "execution_mode": self.execution_mode,
            "requested_gradient_mask": list(self.requested_gradient_mask),
            "gqa_ratio": self.gqa_ratio,
            "tail_class": self.tail_class,
            "workspace_budget_class": self.workspace_budget_class,
            "geometry": self.plan.geometry.to_dict(),
            "planner": {
                "version": self.plan.planner_version,
                "source_hash": self.plan.planner_source_hash,
                "specialization_axes": dict(self.plan.specialization_axes),
                "runtime_axes": dict(self.plan.runtime_axes),
                "canonical_pair_order": dict(self.plan.canonical_pair_order),
            },
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )


def grouped_quadratic_capability_key(
    plan: KernelPlan,
    *,
    execution_mode: Literal[
        "forward",
        "backward",
        "backward_normalization",
        "backward_prefix",
        "backward_suffix",
    ],
) -> GroupedQuadraticCapabilityKey:
    """Build the exact stage key from a live immutable ``KernelPlan``."""
    return GroupedQuadraticCapabilityKey(
        plan=plan,
        execution_mode=execution_mode,
        requested_gradient_mask=plan.geometry.requested_gradient_mask,
    )


def reviewed_record_for_capability_key(
    key: GroupedQuadraticCapabilityKey,
) -> Mapping[str, object] | None:
    """Return the immutable current exact record for ``key``, or ``None``."""
    evidence = reviewed_evidence_for_capability_key(key)
    return evidence.record if evidence.status == "current" else None


def reviewed_evidence_for_capability_key(
    key: GroupedQuadraticCapabilityKey,
    *,
    store: ReviewedRecordStore | None = None,
) -> ReviewedCapabilityEvidence:
    """Report current, stale, or absent evidence without gating execution."""
    if not isinstance(key, GroupedQuadraticCapabilityKey):
        raise TypeError("review lookup requires a GroupedQuadraticCapabilityKey")
    if store is not None:
        resolved_store = store
    elif (
        REVIEWED_GROUPED_QUADRATIC_RECORDS
        is REVIEWED_GROUPED_QUADRATIC_RECORD_STORE.records
    ):
        resolved_store = REVIEWED_GROUPED_QUADRATIC_RECORD_STORE
    else:
        resolved_store = _reviewed_record_store_from_values(
            REVIEWED_GROUPED_QUADRATIC_RECORDS
        )
    expected_key = key.to_dict()
    matching = tuple(
        item
        for item in resolved_store.record_statuses
        if item.status != "invalid"
        and _canonical_json_value(item.record.get("capability_key")) == expected_key
    )
    current = tuple(item for item in matching if item.status == "current")
    diagnostics = list(resolved_store.diagnostics)
    if current:
        canonical_payloads = {canonical_record_payload(item.record) for item in current}
        if len(canonical_payloads) > 1:
            diagnostics.append(
                ReviewedRecordDiagnostic(
                    "conflicting_current",
                    "multiple current records conflict for the exact capability key",
                )
            )
            return ReviewedCapabilityEvidence("unreviewed", None, tuple(diagnostics))
        if len(current) > 1:
            diagnostics.append(
                ReviewedRecordDiagnostic(
                    "duplicate_current",
                    "identical current records duplicate the exact capability key",
                )
            )
        return ReviewedCapabilityEvidence(
            "current", current[0].record, tuple(diagnostics)
        )
    stale = tuple(item for item in matching if item.status == "stale")
    if stale:
        return ReviewedCapabilityEvidence("stale", stale[0].record, tuple(diagnostics))
    return ReviewedCapabilityEvidence("unreviewed", None, tuple(diagnostics))


def capability_key_is_reviewed(key: GroupedQuadraticCapabilityKey) -> bool:
    """Return true only for a digest-valid record with the exact key payload."""
    return reviewed_record_for_capability_key(key) is not None


def require_reviewed_capability_key(key: GroupedQuadraticCapabilityKey) -> None:
    """Validate a capability key while keeping reviewed evidence non-blocking."""
    reviewed_evidence_for_capability_key(key)


def grouped_quadratic_capability_from_tensors(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    linear_coefficients: torch.Tensor,
    *,
    causal: bool,
) -> GroupedQuadraticCapability:
    """Return the retired nine-field geometry tuple for diagnostics only."""
    return (
        q.dtype,
        q.shape[0],
        q.shape[1],
        k.shape[1],
        q.shape[2],
        q.shape[-1],
        v.shape[-1],
        causal,
        linear_coefficients.shape[-1],
    )


__all__ = (
    "GroupedQuadraticCapability",
    "GroupedQuadraticCapabilityKey",
    "REVIEWED_GROUPED_QUADRATIC_CAPABILITIES",
    "REVIEWED_GROUPED_QUADRATIC_RECORD_DIAGNOSTICS",
    "REVIEWED_GROUPED_QUADRATIC_RECORDS",
    "REVIEWED_GROUPED_QUADRATIC_RECORD_STATUSES",
    "ReviewedCapabilityEvidence",
    "ReviewedRecordDiagnostic",
    "ReviewedRecordStatus",
    "ReviewedRecordStore",
    "capability_key_is_reviewed",
    "grouped_quadratic_capability_key",
    "grouped_quadratic_capability_from_tensors",
    "require_reviewed_capability_key",
    "reviewed_evidence_for_capability_key",
    "reviewed_record_for_capability_key",
)
