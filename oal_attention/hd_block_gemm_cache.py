"""Canonical HD pair layouts and process-local device metadata admission."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from threading import Event, Lock
from typing import Any

import torch

from .hd_block_gemm_contracts import (
    _HDParallelBlockPlanContract,
    _MAX_HEAD_DIMENSION,
    _require_plain_int,
    _require_head_dimension,
    _sha256,
    _stable_json,
)
from .hd_block_gemm_profiling import _record_hd_allocation

_INT64_BYTES = 8


def _canonical_runtime_device(device: object, *, name: str) -> torch.device:
    if not isinstance(device, (str, torch.device)):
        raise TypeError(f"{name} must be a device string or torch.device")
    try:
        canonical = torch.device(device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name} is not a valid runtime device") from error
    if canonical.type not in ("cpu", "mps", "cuda"):
        raise ValueError(f"{name} must be CPU, MPS, or CUDA")
    if canonical.type == "cpu":
        if canonical.index is not None or str(canonical) != "cpu":
            raise ValueError(f"{name} CPU device must be exactly 'cpu'")
    elif canonical.index is None:
        raise ValueError(f"{name} accelerator device requires an explicit ordinal")
    expected_identity = (
        "cpu" if canonical.type == "cpu" else f"{canonical.type}:{canonical.index}"
    )
    if str(canonical) != expected_identity:
        raise ValueError(f"{name} must use canonical runtime device spelling")
    if isinstance(device, str) and device != expected_identity:
        raise ValueError(f"{name} must use canonical runtime device spelling")
    return canonical


@dataclass(frozen=True)
class CanonicalPairLayout:
    """Stable value handle for the packed lower-triangular quadratic basis."""

    head_dimension: int
    rows: tuple[int, ...]
    columns: tuple[int, ...]
    multiplicity: tuple[int, ...]
    layout_id: str = ""

    def __post_init__(self) -> None:
        head_dimension = _require_head_dimension(self.head_dimension)
        expected_rows = tuple(
            row for row in range(head_dimension) for _ in range(row + 1)
        )
        expected_columns = tuple(
            column for row in range(head_dimension) for column in range(row + 1)
        )
        expected_multiplicity = tuple(
            1 if row == column else 2
            for row, column in zip(expected_rows, expected_columns, strict=True)
        )
        if self.rows != expected_rows or self.columns != expected_columns:
            raise ValueError(
                "canonical pairs must use lower-triangular row-major order"
            )
        if self.multiplicity != expected_multiplicity:
            raise ValueError(
                "canonical pair multiplicity must be 1 diagonal and 2 off-diagonal"
            )
        payload = {
            "schema": "canonical_hd_pair_layout_v1",
            "head_dimension": head_dimension,
            "rows": list(self.rows),
            "columns": list(self.columns),
            "multiplicity": list(self.multiplicity),
        }
        expected_layout_id = _sha256(payload)
        if self.layout_id and self.layout_id != expected_layout_id:
            raise ValueError("canonical pair layout id does not match its value")
        object.__setattr__(self, "layout_id", expected_layout_id)

    @property
    def pair_count(self) -> int:
        return len(self.rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "head_dimension": self.head_dimension,
            "rows": list(self.rows),
            "columns": list(self.columns),
            "multiplicity": list(self.multiplicity),
            "layout_id": self.layout_id,
        }


@lru_cache(maxsize=None)
def canonical_pair_layout(head_dimension: int) -> CanonicalPairLayout:
    """Return the immutable canonical packed-pair value for a dimension."""
    head_dimension = _require_head_dimension(head_dimension)
    rows = tuple(row for row in range(head_dimension) for _ in range(row + 1))
    columns = tuple(
        column for row in range(head_dimension) for column in range(row + 1)
    )
    multiplicity = tuple(
        1 if row == column else 2 for row, column in zip(rows, columns, strict=True)
    )
    return CanonicalPairLayout(
        head_dimension=head_dimension,
        rows=rows,
        columns=columns,
        multiplicity=multiplicity,
    )


@dataclass(frozen=True)
class PairMetadataCacheEntry:
    """Immutable manifest record for one retained device metadata entry."""

    layout_id: str
    device: str
    pair_count: int
    nbytes: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.layout_id, str)
            or len(self.layout_id) != 64
            or any(character not in "0123456789abcdef" for character in self.layout_id)
        ):
            raise ValueError("cache entry layout_id must be a lowercase SHA256")
        _canonical_runtime_device(self.device, name="cache entry device")
        pair_count = _require_plain_int(self.pair_count, name="pair_count", minimum=1)
        expected_nbytes = 3 * pair_count * _INT64_BYTES
        if self.nbytes and self.nbytes != expected_nbytes:
            raise ValueError("cache entry bytes do not match three int64 pair tensors")
        object.__setattr__(self, "nbytes", expected_nbytes)

    @property
    def key(self) -> tuple[str, str]:
        return (self.layout_id, self.device)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout_id": self.layout_id,
            "device": self.device,
            "pair_count": self.pair_count,
            "nbytes": self.nbytes,
        }


_PAIR_TENSOR_CACHE: dict[
    tuple[str, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
] = {}
_PAIR_TENSOR_CACHE_LOCK = Lock()
_PAIR_CACHE_ACTIVE_LEASES: dict[int, tuple[PairMetadataCacheEntry, ...]] = {}
_PAIR_CACHE_PENDING_MATERIALIZATIONS: dict[tuple[str, str], Event] = {}
_PAIR_CACHE_NEXT_LEASE_TOKEN = 0


@dataclass
class _PairCacheAdmissionLease:
    """Idempotently releasable handle for one admitted cache projection."""

    token: int
    released: bool = False


def _require_plan(plan: object) -> _HDParallelBlockPlanContract:
    if not isinstance(plan, _HDParallelBlockPlanContract):
        raise TypeError("plan must be an HDParallelBlockPlan")
    return plan


def _pair_cache_manifest_locked() -> tuple[PairMetadataCacheEntry, ...]:
    """Describe and validate the complete cache while its lock is held."""
    entries: list[PairMetadataCacheEntry] = []
    for key, tensors in _PAIR_TENSOR_CACHE.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or not isinstance(tensors, tuple)
            or len(tensors) != 3
        ):
            raise RuntimeError("pair metadata cache contains an invalid record")
        layout_id, device_name = key
        rows, columns, multiplicity = tensors
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise RuntimeError("pair metadata cache contains a non-tensor value")
        pair_count = rows.numel()
        expected_shape = (pair_count,)
        if any(
            tensor.shape != expected_shape
            or tensor.dtype != torch.int64
            or str(tensor.device) != device_name
            for tensor in (rows, columns, multiplicity)
        ):
            raise RuntimeError("pair metadata cache tensor contract is corrupted")
        entry = PairMetadataCacheEntry(
            layout_id=layout_id,
            device=device_name,
            pair_count=pair_count,
        )
        if entry.key != key:
            raise RuntimeError("pair metadata cache key is not canonical")
        entries.append(entry)
    return tuple(sorted(entries, key=lambda entry: entry.key))


def _project_cache_manifest(
    before: tuple[PairMetadataCacheEntry, ...],
    current: PairMetadataCacheEntry,
) -> tuple[PairMetadataCacheEntry, ...]:
    by_key = {entry.key: entry for entry in before}
    cached = by_key.get(current.key)
    if cached is not None and cached != current:
        raise RuntimeError("cached pair metadata disagrees with the current layout")
    by_key[current.key] = current
    return tuple(sorted(by_key.values(), key=lambda entry: entry.key))


def _acquire_pair_cache_admission_lease(
    plan: _HDParallelBlockPlanContract,
) -> _PairCacheAdmissionLease:
    """Lease one invocation's exact cache projection without holding the lock."""
    plan = _require_plan(plan)
    # Construct the handle before entering the registry transaction.  In
    # particular, an allocator failure here cannot leave any global state.
    lease = _PairCacheAdmissionLease(token=0)
    global _PAIR_CACHE_NEXT_LEASE_TOKEN
    with _PAIR_TENSOR_CACHE_LOCK:
        actual = _pair_cache_manifest_locked()
        if actual not in (
            plan.cache_before_manifest,
            plan.projected_cache_after_manifest,
        ):
            raise RuntimeError("stale pair metadata cache plan: cache manifest changed")
        if any(
            projected != plan.projected_cache_after_manifest
            for projected in _PAIR_CACHE_ACTIVE_LEASES.values()
        ):
            raise RuntimeError(
                "pair metadata cache admission lease conflicts with an active plan"
            )
        token = _PAIR_CACHE_NEXT_LEASE_TOKEN + 1
        lease.token = token
        _PAIR_CACHE_ACTIVE_LEASES[token] = plan.projected_cache_after_manifest
        _PAIR_CACHE_NEXT_LEASE_TOKEN = token
        return lease


def _validate_pair_cache_admission_lease_locked(
    lease: _PairCacheAdmissionLease,
    plan: _HDParallelBlockPlanContract,
) -> None:
    """Validate a live lease while the caller holds the cache lock."""
    if not isinstance(lease, _PairCacheAdmissionLease):
        raise TypeError("pair cache lease must be an admission lease handle")
    if lease.released:
        raise RuntimeError("pair metadata cache admission lease is not active")
    projected = _PAIR_CACHE_ACTIVE_LEASES.get(lease.token)
    if projected is None:
        raise RuntimeError("pair metadata cache admission lease is not active")
    if projected != plan.projected_cache_after_manifest:
        raise RuntimeError("pair metadata cache admission lease plan mismatch")
    actual = _pair_cache_manifest_locked()
    if actual not in (
        plan.cache_before_manifest,
        plan.projected_cache_after_manifest,
    ):
        raise RuntimeError("stale pair metadata cache plan: cache manifest changed")


def _validate_pair_cache_admission_lease(
    lease: _PairCacheAdmissionLease,
    plan: _HDParallelBlockPlanContract,
) -> None:
    """Validate a live lease before a planned allocation."""
    if not isinstance(lease, _PairCacheAdmissionLease):
        raise TypeError("pair cache lease must be an admission lease handle")
    plan = _require_plan(plan)
    with _PAIR_TENSOR_CACHE_LOCK:
        _validate_pair_cache_admission_lease_locked(lease, plan)


def _release_pair_cache_admission_lease(lease: _PairCacheAdmissionLease) -> None:
    """Release an invocation lease; repeated release is a safe no-op."""
    if not isinstance(lease, _PairCacheAdmissionLease):
        raise TypeError("pair cache lease must be an admission lease handle")
    with _PAIR_TENSOR_CACHE_LOCK:
        if lease.released:
            return
        if _PAIR_CACHE_ACTIVE_LEASES.pop(lease.token, None) is None:
            raise RuntimeError("pair metadata cache admission lease is not active")
        lease.released = True


def _abandon_pair_cache_admission_lease(lease: _PairCacheAdmissionLease) -> None:
    """Best-effort death-owner cleanup that cannot strand cache admission."""
    if not isinstance(lease, _PairCacheAdmissionLease):
        return
    with _PAIR_TENSOR_CACHE_LOCK:
        if lease.released:
            return
        _PAIR_CACHE_ACTIVE_LEASES.pop(lease.token, None)
        lease.released = True


def _materialize_pair_layout(
    layout: CanonicalPairLayout,
    device: torch.device,
    *,
    plan: _HDParallelBlockPlanContract,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize only metadata already admitted by an exact physical plan."""
    if not isinstance(layout, CanonicalPairLayout):
        raise TypeError("layout must be a CanonicalPairLayout")
    plan = _require_plan(plan)
    canonical = canonical_pair_layout(layout.head_dimension)
    if layout != canonical:
        raise ValueError("layout must be the canonical value for its dimension")
    device = _canonical_runtime_device(device, name="materialization device")
    if plan.pair_layout_id != layout.layout_id:
        raise ValueError("plan pair layout does not match requested materialization")
    if plan.device != str(device):
        raise ValueError("plan device does not match requested materialization")
    current_entry = PairMetadataCacheEntry(
        layout_id=layout.layout_id,
        device=str(device),
        pair_count=layout.pair_count,
    )
    if plan.pair_metadata_materialization_bytes != current_entry.nbytes:
        raise ValueError("plan does not account exact pair metadata materialization")
    key = (layout.layout_id, str(device))
    lease = _acquire_pair_cache_admission_lease(plan)
    pending_event: Event | None = None
    owns_materialization = False
    primary_error: BaseException | None = None
    try:
        tensors = None
        while tensors is None and not owns_materialization:
            with _PAIR_TENSOR_CACHE_LOCK:
                _validate_pair_cache_admission_lease_locked(lease, plan)
                tensors = _PAIR_TENSOR_CACHE.get(key)
                if tensors is not None:
                    if (
                        _pair_cache_manifest_locked()
                        != plan.projected_cache_after_manifest
                    ):
                        raise RuntimeError(
                            "pair metadata cache did not reach the planned manifest"
                        )
                    break
                pending_event = _PAIR_CACHE_PENDING_MATERIALIZATIONS.get(key)
                if pending_event is None:
                    pending_event = Event()
                    _PAIR_CACHE_PENDING_MATERIALIZATIONS[key] = pending_event
                    owns_materialization = True
            if not owns_materialization:
                pending_event.wait()

        if owns_materialization:
            allocated = (
                _record_hd_allocation(
                    "pair_rows_cache",
                    torch.tensor(layout.rows, dtype=torch.int64, device=device),
                    category="other",
                ),
                _record_hd_allocation(
                    "pair_columns_cache",
                    torch.tensor(layout.columns, dtype=torch.int64, device=device),
                    category="other",
                ),
                _record_hd_allocation(
                    "pair_multiplicity_cache",
                    torch.tensor(layout.multiplicity, dtype=torch.int64, device=device),
                    category="other",
                ),
            )
            try:
                allocated_devices = tuple(
                    _canonical_runtime_device(
                        tensor.device,
                        name="allocated pair metadata tensor device",
                    )
                    for tensor in allocated
                )
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    "allocated pair metadata tensor has an invalid device identity"
                ) from error
            if any(
                allocated_device != device for allocated_device in allocated_devices
            ):
                raise RuntimeError(
                    "allocated pair metadata tensor device does not match the plan"
                )
            with _PAIR_TENSOR_CACHE_LOCK:
                _validate_pair_cache_admission_lease_locked(lease, plan)
                tensors = _PAIR_TENSOR_CACHE.get(key)
                if tensors is None:
                    tensors = allocated
                    _PAIR_TENSOR_CACHE[key] = tensors
                actual_after = _pair_cache_manifest_locked()
                if actual_after != plan.projected_cache_after_manifest:
                    raise RuntimeError(
                        "pair metadata cache did not reach the planned manifest"
                    )
                registered = _PAIR_CACHE_PENDING_MATERIALIZATIONS.pop(key, None)
                if registered is not pending_event:
                    raise RuntimeError(
                        "pair metadata materialization reservation was lost"
                    )
                pending_event.set()
                owns_materialization = False

        if tensors is None:
            raise RuntimeError("pair metadata materialization produced no tensors")
        return (
            _record_hd_allocation(
                "pair_rows_materialized",
                tensors[0].clone(),
                category="other",
            ),
            _record_hd_allocation(
                "pair_columns_materialized",
                tensors[1].clone(),
                category="other",
            ),
            _record_hd_allocation(
                "pair_multiplicity_materialized",
                tensors[2].clone(),
                category="other",
            ),
        )
    except BaseException as error:
        primary_error = error
        if owns_materialization and pending_event is not None:
            with _PAIR_TENSOR_CACHE_LOCK:
                registered = _PAIR_CACHE_PENDING_MATERIALIZATIONS.get(key)
                if registered is pending_event:
                    del _PAIR_CACHE_PENDING_MATERIALIZATIONS[key]
                    pending_event.set()
        raise
    finally:
        try:
            _release_pair_cache_admission_lease(lease)
        except BaseException:
            _abandon_pair_cache_admission_lease(lease)
            if primary_error is None:
                raise


__all__: tuple[str, ...] = ()
