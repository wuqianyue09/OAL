"""Shared smoke tensors, parameter-update audits, and resource evidence."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import time
from .workflows.errors import OrchestrationError


@dataclass(frozen=True)
class ProductionSmokeRuntime:
    """The lazily assembled objects used only by a normal smoke invocation."""

    bundle: object
    assets: object


def _smoke_input_ids(*, assets: object, sequence_length: int, device: object) -> object:
    """Return one canonical block at exactly the requested smoke length."""
    import torch

    train_blocks = getattr(assets, "train_blocks", None)
    if train_blocks is None or len(train_blocks) < 1:
        raise OrchestrationError("production smoke assets have no train blocks")
    block = torch.as_tensor(train_blocks[0], device=device, dtype=torch.long)
    if block.ndim != 1 or block.shape[0] < sequence_length:
        raise OrchestrationError(
            "production smoke train block must be rank-1 and at least the requested sequence length"
        )
    return block[:sequence_length].contiguous().unsqueeze(0)


def _json_safe_smoke_scalar(value: object) -> float | None:
    """Return one finite scalar for JSON evidence, otherwise ``None``."""
    import math
    import torch

    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        return None
    scalar = float(value.detach().item())
    return scalar if math.isfinite(scalar) else None


def _finalize_smoke_resources(
    production_device: object, *, measures_cuda: bool, started_at: float
) -> dict[str, object]:
    """Finish two-step timing and peak-memory evidence on success or failure."""
    import math
    import torch

    if measures_cuda:
        torch.cuda.synchronize(production_device)
    elapsed_seconds = time.perf_counter() - started_at
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
        raise OrchestrationError("full-model smoke measured an invalid elapsed time")
    if not measures_cuda:
        return {
            "elapsed_seconds": elapsed_seconds,
            "memory": {
                "applicability": "not_applicable",
                "device": str(production_device),
                "reason": "peak CUDA memory is not applicable to a CPU production device",
            },
        }
    peak_allocated_bytes = int(torch.cuda.max_memory_allocated(production_device))
    peak_reserved_bytes = int(torch.cuda.max_memory_reserved(production_device))
    if peak_allocated_bytes < 0 or peak_reserved_bytes < 0:
        raise OrchestrationError("full-model smoke measured invalid CUDA peak memory")
    return {
        "elapsed_seconds": elapsed_seconds,
        "memory": {
            "applicability": "applicable",
            "device": str(production_device),
            "peak_allocated_bytes": peak_allocated_bytes,
            "peak_reserved_bytes": peak_reserved_bytes,
        },
    }


def _require_smoke_gradients(
    parameters: Mapping[str, object],
    *,
    category: str,
    optimizer_step: int,
    require_nonzero: bool = False,
) -> None:
    """Require finite first-order gradients at the current two-step gate point."""
    import torch

    for name, parameter in parameters.items():
        gradient = getattr(parameter, "grad", None)
        if not isinstance(gradient, torch.Tensor):
            raise OrchestrationError(
                f"{category} parameter {name} has no gradient at optimizer step {optimizer_step}"
            )
        if not bool(torch.isfinite(gradient).all()):
            raise OrchestrationError(
                f"{category} parameter {name} has non-finite gradient at optimizer step {optimizer_step}"
            )
        if require_nonzero and float(gradient.detach().float().norm().item()) <= 0.0:
            raise OrchestrationError(
                f"{category} parameter {name} has zero gradient at optimizer step {optimizer_step}"
            )


def _require_smoke_updates(
    records: Sequence[Mapping[str, object]], *, category: str, optimizer_step: int
) -> None:
    """Require every audited parameter to change during this exact smoke step."""
    for record in records:
        if record.get("updated") is not True:
            raise OrchestrationError(
                f"{category} parameter {record.get('name')} did not update at optimizer step {optimizer_step}"
            )


def _smoke_kernel_audit(
    parameters: Mapping[str, object],
    before: Mapping[str, object],
    *,
    prefix: str,
    method: str,
    strict: bool,
    require_nonzero: bool = False,
) -> dict[str, object]:
    """Represent method-validated kernel evidence with stable applicability."""
    if not parameters:
        return {
            "applicability": "not_applicable",
            "reason": f"{method} has no trainable kernel parameter",
            "parameters": [],
        }
    records = (
        _smoke_update_audit(
            parameters, before, prefix=prefix, include_relative_update=require_nonzero
        )
        if strict
        else _json_safe_smoke_update_audit(parameters, before, prefix=prefix)
    )
    return {"applicability": "applicable", "parameters": records}


def _smoke_update_audit(
    parameters: Mapping[str, object],
    before: Mapping[str, object],
    *,
    prefix: str,
    include_relative_update: bool = False,
) -> list[dict[str, object]]:
    """Record finite gradient and update evidence without serializing tensors."""
    import math
    import torch

    records: list[dict[str, object]] = []
    for name, parameter in sorted(parameters.items()):
        prior = before[prefix + name]
        current = getattr(parameter, "detach")()
        gradient = getattr(parameter, "grad", None)
        if not isinstance(prior, torch.Tensor) or not isinstance(current, torch.Tensor):
            raise OrchestrationError(f"smoke audit parameter {name} is not a Tensor")
        if not isinstance(gradient, torch.Tensor):
            raise OrchestrationError(f"smoke audit parameter {name} has no gradient")
        update = current - prior
        gradient_norm = float(gradient.detach().float().norm().item())
        update_norm = float(update.float().norm().item())
        if not math.isfinite(gradient_norm) or not math.isfinite(update_norm):
            raise OrchestrationError(
                f"smoke audit parameter {name} has non-finite norm"
            )
        if not bool(torch.isfinite(current).all()) or not bool(
            torch.isfinite(update).all()
        ):
            raise OrchestrationError(
                f"smoke audit parameter {name} has non-finite update"
            )
        record: dict[str, object] = {
            "name": name,
            "dtype": str(current.dtype),
            "grad_norm": gradient_norm,
            "update_norm": update_norm,
            "finite_grad": bool(torch.isfinite(gradient).all()),
            "finite_parameter": bool(torch.isfinite(current).all()),
            "finite_update": bool(torch.isfinite(update).all()),
            "updated": not torch.equal(current, prior),
        }
        if include_relative_update:
            prior_norm = float(prior.float().norm().item())
            relative_update = update_norm / max(prior_norm, 1e-12)
            record.update(
                {
                    "relative_update": relative_update,
                    "relative_update_finite": math.isfinite(relative_update),
                }
            )
        records.append(record)
    return records


def _json_safe_smoke_update_audit(
    parameters: Mapping[str, object], before: Mapping[str, object], *, prefix: str
) -> list[dict[str, object]]:
    """Capture partial failure evidence without emitting NaN/Inf JSON numbers."""
    import math
    import torch

    records: list[dict[str, object]] = []
    for name, parameter in sorted(parameters.items()):
        prior = before.get(prefix + name)
        current = getattr(parameter, "detach", lambda: None)()
        gradient = getattr(parameter, "grad", None)
        finite_gradient = isinstance(gradient, torch.Tensor) and bool(
            torch.isfinite(gradient).all()
        )
        gradient_norm: float | None = None
        if finite_gradient:
            candidate_gradient_norm = float(gradient.detach().float().norm().item())
            if math.isfinite(candidate_gradient_norm):
                gradient_norm = candidate_gradient_norm
        finite_parameter = isinstance(current, torch.Tensor) and bool(
            torch.isfinite(current).all()
        )
        update: object = None
        if isinstance(prior, torch.Tensor) and isinstance(current, torch.Tensor):
            try:
                update = current - prior
            except (RuntimeError, TypeError):
                update = None
        finite_update = isinstance(update, torch.Tensor) and bool(
            torch.isfinite(update).all()
        )
        update_norm: float | None = None
        if finite_update:
            candidate_update_norm = float(update.float().norm().item())
            if math.isfinite(candidate_update_norm):
                update_norm = candidate_update_norm
        updated = bool(
            finite_parameter
            and finite_update
            and isinstance(prior, torch.Tensor)
            and isinstance(current, torch.Tensor)
            and (not torch.equal(current, prior))
        )
        records.append(
            {
                "name": name,
                "dtype": (
                    str(current.dtype) if isinstance(current, torch.Tensor) else None
                ),
                "grad_norm": gradient_norm,
                "update_norm": update_norm,
                "finite_grad": finite_gradient and gradient_norm is not None,
                "finite_parameter": finite_parameter,
                "finite_update": finite_update and update_norm is not None,
                "updated": updated,
            }
        )
    return records
