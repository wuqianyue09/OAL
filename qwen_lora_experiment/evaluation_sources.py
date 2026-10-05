"""Load the trained OAL LoRA adapter selected by validation NLL."""

from __future__ import annotations
from collections.abc import Mapping
import math
from pathlib import Path
from .checkpointing import CheckpointContext
from .paths import sha256_file
from .evaluation_contracts import EvaluationContext, _BestLoader, _json_object


def _checkpoint_evidence(loaded: object) -> dict[str, object]:
    path = getattr(loaded, "path", None)
    step = getattr(loaded, "step", None)
    validation_nll = getattr(loaded, "validation_nll", None)
    if not isinstance(path, Path) or not path.is_file():
        raise ValueError("best_loader must return an existing best checkpoint path")
    if type(step) is not int or step < 0:
        raise ValueError("best_loader returned an invalid checkpoint step")
    if not isinstance(validation_nll, (int, float)) or isinstance(validation_nll, bool):
        raise ValueError("best_loader returned an invalid checkpoint validation_nll")
    resolved_nll = float(validation_nll)
    if not math.isfinite(resolved_nll):
        raise ValueError("best_loader returned a non-finite checkpoint validation_nll")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "step": step,
        "validation_nll": resolved_nll,
    }


def _evaluation_source_kind(checkpoint_context: EvaluationContext) -> str:
    if not isinstance(checkpoint_context, CheckpointContext):
        raise TypeError("checkpoint_context must be a LoRA CheckpointContext")
    return "best_adapter"


def _load_evaluation_source(
    run_dir: Path,
    *,
    model: object,
    kernel_bank: object | None,
    checkpoint_context: EvaluationContext,
    best_loader: _BestLoader | None,
) -> dict[str, object]:
    _evaluation_source_kind(checkpoint_context)
    if not callable(best_loader):
        raise TypeError("checkpoint-backed evaluation requires a callable best_loader")
    loaded = best_loader(
        run_dir, model=model, kernel_bank=kernel_bank, context=checkpoint_context
    )
    evidence = _checkpoint_evidence(loaded)
    if getattr(loaded, "method", None) != checkpoint_context.method:
        raise ValueError("best_loader returned a checkpoint for the wrong method")
    return evidence


def _require_same_evaluation_source(
    nll_record: Mapping[str, object], *, evaluation_source: Mapping[str, object]
) -> None:
    if nll_record.get("best_checkpoint") != _json_object(
        evaluation_source, "best_checkpoint"
    ):
        raise ValueError("NLL and PIQA evaluation sources do not match")
    if "evaluation_source" in nll_record:
        raise ValueError("NLL evaluation source uses an incompatible legacy field")
