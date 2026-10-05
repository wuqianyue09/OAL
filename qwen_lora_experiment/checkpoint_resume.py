"""Transactional application of resume state with runtime rollback."""

from __future__ import annotations
from collections.abc import Mapping
import copy
from dataclasses import dataclass
from torch import Tensor
from torch.optim import Optimizer
from .checkpoint_state import (
    _copy_actions,
    _load_scheduler_state,
    _capture_rng_state,
    _restore_rng_state,
    _move_optimizer_state_to_owner_devices,
)


@dataclass
class _ResumeRuntimeSnapshot:
    """Private rollback state captured before any latest-resume application."""

    core_tensors: list[tuple[Tensor, Tensor]]
    optimizer_attributes: dict[str, object]
    scheduler_attributes: dict[str, object] | None
    object_identity_memo: dict[int, object]
    rng_state: dict[str, object]


def _apply_resume_state(
    payload: Mapping[str, object],
    *,
    actions: list[tuple[Tensor, Tensor]],
    optimizer: Optimizer,
    scheduler: object | None,
) -> None:
    """Install validated LoRA and OAL state, rolling back every owner on failure."""
    snapshot = _capture_resume_runtime_snapshot(
        actions=actions, optimizer=optimizer, scheduler=scheduler
    )
    try:
        _preflight_resume_application(
            optimizer_state=payload["optimizer_state"],
            scheduler_state=payload["scheduler_state"],
            rng_state=payload["rng_state"],
            optimizer=optimizer,
            scheduler=scheduler,
            snapshot=snapshot,
        )
        _copy_actions(actions)
        _load_optimizer_state(payload["optimizer_state"], optimizer)
        _load_scheduler_state(payload["scheduler_state"], scheduler)
        _restore_rng_state(payload["rng_state"])
    except BaseException:
        _rollback_resume_runtime(
            snapshot=snapshot, optimizer=optimizer, scheduler=scheduler
        )
        raise


def _capture_resume_runtime_snapshot(
    *,
    actions: list[tuple[Tensor, Tensor]],
    optimizer: Optimizer,
    scheduler: object | None,
) -> _ResumeRuntimeSnapshot:
    """Capture every mutable participant before application or semantic probing.

    A protocol implementation may mutate then raise from ``load_state_dict``.
    We therefore retain an object-attribute snapshot in addition to normal
    ``state_dict`` validation, rather than trusting a failing protocol method
    to restore itself.
    """
    identity_memo = _resume_object_identity_memo(optimizer)
    core_tensors = [(target, target.detach().clone()) for (target, _) in actions]
    optimizer_attributes = _snapshot_object_attributes(
        optimizer, "optimizer", memo=identity_memo
    )
    scheduler_attributes = (
        None
        if scheduler is None
        else _snapshot_object_attributes(scheduler, "scheduler", memo=identity_memo)
    )
    return _ResumeRuntimeSnapshot(
        core_tensors=core_tensors,
        optimizer_attributes=optimizer_attributes,
        scheduler_attributes=scheduler_attributes,
        object_identity_memo=dict(identity_memo),
        rng_state=_capture_rng_state(),
    )


def _resume_object_identity_memo(optimizer: Optimizer) -> dict[int, object]:
    """Keep live optimizer/Parameter identities across attribute snapshot clones."""
    memo = {
        id(parameter): parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    memo[id(optimizer)] = optimizer
    return memo


def _snapshot_object_attributes(
    value: object, label: str, *, memo: dict[int, object]
) -> dict[str, object]:
    """Deep-copy a mutable protocol object's attributes while preserving parameters."""
    try:
        attributes = vars(value)
    except TypeError as exc:
        raise TypeError(
            f"{label} must expose mutable __dict__ state for failure-atomic resume"
        ) from exc
    try:
        snapshot = copy.deepcopy(dict(attributes), memo=dict(memo))
    except Exception as exc:
        raise ValueError(
            f"{label} runtime state cannot be snapshotted for failure-atomic resume: {exc}"
        ) from exc
    if not isinstance(snapshot, dict):
        raise ValueError(f"{label} runtime attribute snapshot must be a dictionary")
    return snapshot


def _restore_object_attributes(
    value: object,
    snapshot: Mapping[str, object],
    label: str,
    *,
    memo: dict[int, object],
) -> None:
    """Restore a fresh clone without invoking a potentially failing protocol."""
    try:
        attributes = vars(value)
    except TypeError as exc:
        raise RuntimeError(
            f"{label} lost mutable __dict__ state during resume rollback"
        ) from exc
    try:
        restored = copy.deepcopy(dict(snapshot), memo=dict(memo))
    except Exception as exc:
        raise RuntimeError(
            f"{label} rollback could not clone runtime attributes: {exc}"
        ) from exc
    try:
        attributes.clear()
        attributes.update(restored)
    except Exception as exc:
        raise RuntimeError(
            f"{label} rollback could not restore runtime attributes: {exc}"
        ) from exc


def _preflight_resume_application(
    *,
    optimizer_state: object,
    scheduler_state: object,
    rng_state: object,
    optimizer: Optimizer,
    scheduler: object | None,
    snapshot: _ResumeRuntimeSnapshot,
) -> None:
    """Exercise semantic protocol paths, restoring the snapshot after each probe."""
    try:
        _restore_rng_state(rng_state)
    finally:
        _restore_rng_state(snapshot.rng_state)
    try:
        _load_optimizer_state(optimizer_state, optimizer)
    finally:
        _restore_object_attributes(
            optimizer,
            snapshot.optimizer_attributes,
            "optimizer",
            memo=snapshot.object_identity_memo,
        )
    try:
        _load_scheduler_state(scheduler_state, scheduler)
    finally:
        if scheduler is not None:
            assert snapshot.scheduler_attributes is not None
            _restore_object_attributes(
                scheduler,
                snapshot.scheduler_attributes,
                "scheduler",
                memo=snapshot.object_identity_memo,
            )
        _restore_rng_state(snapshot.rng_state)


def _rollback_resume_runtime(
    *, snapshot: _ResumeRuntimeSnapshot, optimizer: Optimizer, scheduler: object | None
) -> None:
    """Rollback all mutable participants after any resume application failure."""
    rollback_errors: list[str] = []
    try:
        _copy_actions(snapshot.core_tensors)
    except Exception as exc:
        rollback_errors.append(f"core tensors: {type(exc).__name__}: {exc}")
    try:
        _restore_object_attributes(
            optimizer,
            snapshot.optimizer_attributes,
            "optimizer",
            memo=snapshot.object_identity_memo,
        )
    except Exception as exc:
        rollback_errors.append(f"optimizer: {type(exc).__name__}: {exc}")
    if scheduler is not None:
        try:
            assert snapshot.scheduler_attributes is not None
            _restore_object_attributes(
                scheduler,
                snapshot.scheduler_attributes,
                "scheduler",
                memo=snapshot.object_identity_memo,
            )
        except Exception as exc:
            rollback_errors.append(f"scheduler: {type(exc).__name__}: {exc}")
    try:
        _restore_rng_state(snapshot.rng_state)
    except Exception as exc:
        rollback_errors.append(f"RNG: {type(exc).__name__}: {exc}")
    if rollback_errors:
        raise RuntimeError(
            "latest resume failed and rollback was incomplete: "
            + "; ".join(rollback_errors)
        )


def _load_optimizer_state(payload_state: object, optimizer: Optimizer) -> None:
    try:
        optimizer.load_state_dict(payload_state)
    except Exception as exc:
        raise ValueError(
            f"optimizer_state is incompatible with the supplied optimizer: {exc}"
        ) from exc
    _move_optimizer_state_to_owner_devices(optimizer)
