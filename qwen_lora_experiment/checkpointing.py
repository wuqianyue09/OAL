"""OAL + LoRA checkpoint save/load entry points. Base model weights are never serialized."""

from __future__ import annotations
from .checkpoint_context import add_model_scope_identity
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from torch import Tensor, nn
from torch.optim import Optimizer
from .experiment_contract import BEST_ADAPTER_FILENAME
from .kernel_parameters import KernelParameterBank
from .paths import atomic_torch_save
from .telemetry import require_cohort_member_mutable, cohort_member_lock
from .checkpoint_context import (
    LATEST_RESUME_FILENAME,
    CheckpointContext,
    _normalise_metadata_mapping,
)
from .checkpoint_optimizer import (
    _validate_optimizer_ownership,
    _build_optimizer_bindings,
)
from .checkpoint_payload import (
    _CORE_FIELDS,
    _build_core_payload,
    _validate_core_payload,
    _validate_resume_payload,
)
from .checkpoint_resume import _apply_resume_state
from .checkpoint_state import (
    _copy_actions,
    _checkpoint_path,
    _safe_load,
    _require_payload_mapping,
    _payload_step,
    _payload_nll,
    _clone_checkpoint_tree,
    _scheduler_state_for_save,
    _capture_rng_state,
)


@dataclass(frozen=True)
class BestCheckpointResult:
    """Result of attempting to update the rolling best-adapter file."""

    path: Path
    saved: bool
    validation_nll: float
    previous_best_nll: float | None


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Evidence returned after a validated best-adapter load."""

    path: Path
    step: int
    validation_nll: float
    method: str


@dataclass(frozen=True)
class ResumeCheckpoint(LoadedCheckpoint):
    """Additional evidence returned after a fully restored resume checkpoint."""

    cursor: dict[str, object]
    rng_restored: bool


def save_best_adapter(
    run_dir: str | Path,
    *,
    model: nn.Module,
    kernel_bank: KernelParameterBank,
    context: CheckpointContext,
    step: int,
    validation_nll: float,
) -> BestCheckpointResult:
    """Write ``best_adapter.pt`` only for a strictly lower finite validation NLL.

    The existing file, if present, is safe-loaded and schema/identity-checked
    before its metric is consulted.  A tied metric never overwrites it.
    """
    destination = _checkpoint_path(run_dir, BEST_ADAPTER_FILENAME)
    with cohort_member_lock(destination.parent):
        require_cohort_member_mutable(destination.parent)
        payload = _build_core_payload(
            kind="best_adapter",
            model=model,
            kernel_bank=kernel_bank,
            context=context,
            step=step,
            validation_nll=validation_nll,
        )
        previous_best_nll: float | None = None
        if destination.exists():
            existing = _safe_load(destination)
            _validate_core_payload(
                existing,
                expected_kind="best_adapter",
                model=model,
                kernel_bank=kernel_bank,
                context=context,
            )
            previous_best_nll = _payload_nll(existing)
            if payload["validation_nll"] >= previous_best_nll:
                return BestCheckpointResult(
                    path=destination,
                    saved=False,
                    validation_nll=payload["validation_nll"],
                    previous_best_nll=previous_best_nll,
                )
        atomic_torch_save(destination, payload)
        return BestCheckpointResult(
            path=destination,
            saved=True,
            validation_nll=payload["validation_nll"],
            previous_best_nll=previous_best_nll,
        )


def save_latest_resume(
    run_dir: str | Path,
    *,
    model: nn.Module,
    kernel_bank: KernelParameterBank,
    context: CheckpointContext,
    step: int,
    validation_nll: float,
    optimizer: Optimizer,
    scheduler: object | None,
    cursor: Mapping[str, object],
) -> Path:
    """Atomically replace the one rolling, exact-resume checkpoint.

    ``optimizer`` is required because a latest checkpoint without optimizer
    moments is not an exact resume.  A scheduler may be ``None``; that fact is
    persisted explicitly and must match on restoration.
    """
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer")
    _validate_optimizer_ownership(
        model=model, kernel_bank=kernel_bank, optimizer=optimizer
    )
    payload = _build_core_payload(
        kind="latest_resume",
        model=model,
        kernel_bank=kernel_bank,
        context=context,
        step=step,
        validation_nll=validation_nll,
    )
    optimizer_state = _clone_checkpoint_tree(optimizer.state_dict(), "optimizer_state")
    if not isinstance(optimizer_state, dict):
        raise ValueError("optimizer.state_dict() must return a dictionary")
    payload.update(
        {
            "optimizer_state": optimizer_state,
            "optimizer_bindings": _build_optimizer_bindings(
                model=model,
                kernel_bank=kernel_bank,
                optimizer=optimizer,
                optimizer_state=optimizer_state,
            ),
            "scheduler_state": (
                None if scheduler is None else _scheduler_state_for_save(scheduler)
            ),
            "rng_state": _capture_rng_state(),
            "cursor": _normalise_metadata_mapping(cursor, "cursor"),
        }
    )
    _validate_resume_payload(
        payload,
        model=model,
        kernel_bank=kernel_bank,
        context=context,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    destination = _checkpoint_path(run_dir, LATEST_RESUME_FILENAME)
    with cohort_member_lock(destination.parent):
        require_cohort_member_mutable(destination.parent)
        atomic_torch_save(destination, payload)
    return destination


def load_best_adapter(
    run_dir: str | Path,
    *,
    model: nn.Module,
    kernel_bank: KernelParameterBank,
    context: CheckpointContext,
) -> LoadedCheckpoint:
    """Load the best adapter/kernel state by in-place tensor ``copy_`` only."""
    path, payload, actions = _best_adapter_copy_plan(
        run_dir, model=model, kernel_bank=kernel_bank, context=context
    )
    _copy_actions(actions)
    return LoadedCheckpoint(
        path=path,
        step=_payload_step(payload),
        validation_nll=_payload_nll(payload),
        method=context.method,
    )


def preflight_best_adapter(
    run_dir: str | Path,
    *,
    model: nn.Module,
    kernel_bank: KernelParameterBank,
    context: CheckpointContext,
) -> LoadedCheckpoint:
    """Validate the complete best-adapter copy plan without mutating the runtime."""
    path, payload, _ = _best_adapter_copy_plan(
        run_dir, model=model, kernel_bank=kernel_bank, context=context
    )
    return LoadedCheckpoint(
        path=path,
        step=_payload_step(payload),
        validation_nll=_payload_nll(payload),
        method=context.method,
    )


def _best_adapter_copy_plan(
    run_dir: str | Path,
    *,
    model: nn.Module,
    kernel_bank: KernelParameterBank,
    context: CheckpointContext,
) -> tuple[Path, Mapping[str, object], list[tuple[Tensor, Tensor]]]:
    path = _checkpoint_path(run_dir, BEST_ADAPTER_FILENAME, require_file=True)
    payload = _safe_load(path)
    actions = _validate_core_payload(
        payload,
        expected_kind="best_adapter",
        model=model,
        kernel_bank=kernel_bank,
        context=context,
        allow_historical_grouped_identity=True,
    )
    record = _require_payload_mapping(
        payload, expected_fields=_CORE_FIELDS, label="checkpoint"
    )
    return (path, record, actions)


def load_latest_resume(
    run_dir: str | Path,
    *,
    model: nn.Module,
    kernel_bank: KernelParameterBank,
    context: CheckpointContext,
    optimizer: Optimizer,
    scheduler: object | None,
) -> ResumeCheckpoint:
    """Restore one latest checkpoint after strict schema and identity checks.

    Core tensors are validated then copied into their already-registered
    parameters/buffers.  Optimizer state is loaded separately and every state
    tensor is moved to the device of its owning parameter afterwards.
    """
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer")
    _validate_optimizer_ownership(
        model=model, kernel_bank=kernel_bank, optimizer=optimizer
    )
    path = _checkpoint_path(run_dir, LATEST_RESUME_FILENAME, require_file=True)
    payload = _safe_load(path)
    actions = _validate_resume_payload(
        payload,
        model=model,
        kernel_bank=kernel_bank,
        context=context,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    _apply_resume_state(
        payload, actions=actions, optimizer=optimizer, scheduler=scheduler
    )
    cursor = _normalise_metadata_mapping(payload["cursor"], "cursor")
    return ResumeCheckpoint(
        path=path,
        step=_payload_step(payload),
        validation_nll=_payload_nll(payload),
        method=context.method,
        cursor=cursor,
        rng_restored=True,
    )
