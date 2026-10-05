"""Gradient, update and early-step contracts shared by training modes.

The training loop supplies snapshots and pre-clip evidence explicitly; this
module owns their interpretation without depending on loop orchestration.
"""

from __future__ import annotations
from collections.abc import Iterable, Iterator, Mapping, Sequence
import math
import torch
from torch import Tensor, nn
from .lora import ADAPTER_MODULE_NAME

EARLY_UPDATE_DIAGNOSTIC_STEPS = 10


class TrainingFailure(RuntimeError):
    """A contract violation annotated for durable failure telemetry."""

    def __init__(
        self,
        message: str,
        *,
        phase: str,
        step: int,
        parameter: str | None = None,
        value: object | None = None,
    ) -> None:
        super().__init__(message)
        self.phase = phase
        self.step = step
        self.parameter = parameter
        self.value = value


def require_finite_gradients(
    trainables: Mapping[str, nn.Parameter], *, step: int
) -> None:
    gradients: list[Tensor] = []
    for name, parameter in trainables.items():
        gradient = parameter.grad
        if gradient is None:
            raise TrainingFailure(
                "trainable parameter has no gradient",
                phase="backward",
                step=step,
                parameter=name,
            )
        gradients.append(gradient)
    if _all_tensors_finite(gradients):
        return
    for name, parameter in trainables.items():
        gradient = parameter.grad
        assert gradient is not None
        if not bool(torch.isfinite(gradient).all()):
            raise TrainingFailure(
                "trainable parameter has a non-finite gradient",
                phase="backward",
                step=step,
                parameter=name,
                value=float(gradient.detach().abs().max().item()),
            )
    raise AssertionError("finite-gradient aggregate failed without a failing gradient")


def _all_tensors_finite(tensors: Iterable[Tensor]) -> bool:
    """Aggregate finite predicates locally on each device before synchronizing."""
    predicates_by_device: dict[torch.device, Tensor] = {}
    for tensor in tensors:
        predicate = torch.isfinite(tensor).all()
        existing = predicates_by_device.get(tensor.device)
        predicates_by_device[tensor.device] = (
            predicate if existing is None else torch.logical_and(existing, predicate)
        )
    if not predicates_by_device:
        raise ValueError("finite check requires at least one tensor")
    return all((bool(predicate) for predicate in predicates_by_device.values()))


def pre_clip_gradient_norms(trainables: Mapping[str, nn.Parameter]) -> dict[str, float]:
    """Capture the kernel tuning connectivity signal before gradient clipping mutates it."""
    norms: dict[str, float] = {}
    for name, parameter in trainables.items():
        gradient = parameter.grad
        assert gradient is not None
        norm = float(gradient.detach().float().norm().item())
        if not math.isfinite(norm):
            raise AssertionError(
                "finite gradient validation accepted a non-finite norm"
            )
        norms[name] = norm
    return norms


def parameter_diagnostics(
    trainables: Mapping[str, nn.Parameter],
    before: Mapping[str, Tensor],
    *,
    pre_clip_gradient_norms: Mapping[str, float],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if set(pre_clip_gradient_norms) != set(trainables):
        raise RuntimeError(
            "early parameter diagnostics require complete pre-clip gradient norms"
        )
    for name, parameter in trainables.items():
        prior = before[name]
        gradient = parameter.grad
        assert gradient is not None
        update = parameter.detach() - prior
        parameter_norm = float(parameter.detach().float().norm().item())
        update_norm = float(update.float().norm().item())
        relative_update = update_norm / max(float(prior.float().norm().item()), 1e-12)
        grad_norm = pre_clip_gradient_norms[name]
        records.append(
            {
                "name": name,
                "grad_norm": grad_norm,
                "grad_norm_finite": math.isfinite(grad_norm),
                "parameter_norm": parameter_norm,
                "parameter_norm_finite": math.isfinite(parameter_norm),
                "update_norm": update_norm,
                "update_norm_finite": math.isfinite(update_norm),
                "relative_update": relative_update,
                "relative_update_finite": math.isfinite(relative_update),
                "finite_grad": bool(torch.isfinite(gradient).all()),
                "parameter_finite": bool(torch.isfinite(parameter.detach()).all()),
                "update_finite": bool(torch.isfinite(update).all()),
                "updated": not torch.equal(parameter.detach(), prior),
            }
        )
    return records


def require_finite_parameter_state_after_step(
    trainables: Mapping[str, nn.Parameter], before: Mapping[str, Tensor], *, step: int
) -> None:
    """Fail before telemetry or validation if an optimizer corrupts its state."""
    if not _all_tensors_finite(_iter_parameter_state_tensors(trainables, before)):
        _locate_nonfinite_parameter_state(
            trainables,
            before,
            step=step,
            include_norms=step <= EARLY_UPDATE_DIAGNOSTIC_STEPS,
        )
        raise AssertionError(
            "finite parameter-state aggregate failed without a failing parameter"
        )
    if step <= EARLY_UPDATE_DIAGNOSTIC_STEPS:
        require_finite_parameter_update_norms(trainables, before, step=step)


def _iter_parameter_state_tensors(
    trainables: Mapping[str, nn.Parameter], before: Mapping[str, Tensor]
) -> Iterator[Tensor]:
    """Yield current values and exact updates without retaining all updates at once."""
    for name, parameter in trainables.items():
        yield parameter.detach()
        yield (parameter.detach() - before[name])


def _locate_nonfinite_parameter_state(
    trainables: Mapping[str, nn.Parameter],
    before: Mapping[str, Tensor],
    *,
    step: int,
    include_norms: bool,
) -> None:
    """Preserve actionable parameter context after a failed aggregate check."""
    for name, parameter in trainables.items():
        current = parameter.detach()
        if not bool(torch.isfinite(current).all()):
            raise TrainingFailure(
                "non-finite parameter after optimizer step",
                phase="optimizer_step",
                step=step,
                parameter=name,
            )
        update = current - before[name]
        if not bool(torch.isfinite(update).all()):
            raise TrainingFailure(
                "non-finite parameter update after optimizer step",
                phase="optimizer_step",
                step=step,
                parameter=name,
            )
        if include_norms:
            require_finite_parameter_update_norms(
                {name: parameter}, {name: before[name]}, step=step
            )


def require_finite_parameter_update_norms(
    trainables: Mapping[str, nn.Parameter], before: Mapping[str, Tensor], *, step: int
) -> None:
    """Keep the complete first-ten-step norm contract out of the steady state."""
    for name, parameter in trainables.items():
        current = parameter.detach()
        update = current - before[name]
        parameter_norm = float(current.float().norm().item())
        update_norm = float(update.float().norm().item())
        prior_norm = float(before[name].float().norm().item())
        relative_update = update_norm / max(prior_norm, 1e-12)
        if not all(
            (
                math.isfinite(value)
                for value in (parameter_norm, update_norm, prior_norm, relative_update)
            )
        ):
            raise TrainingFailure(
                "non-finite parameter/update norm after optimizer step",
                phase="optimizer_step",
                step=step,
                parameter=name,
            )


def enforce_early_update_contract(
    diagnostics: Sequence[Mapping[str, object]],
    *,
    step: int,
    expected_lora_b_names: Sequence[str],
) -> None:
    if step > EARLY_UPDATE_DIAGNOSTIC_STEPS:
        return
    records = {record["name"]: record for record in diagnostics}
    if step == 1:
        mandatory = [*expected_lora_b_names]
        mandatory.extend((name for name in records if name.startswith("kernel_bank.")))
        for name in mandatory:
            record = records[name]
            if record["finite_grad"] is not True or record["updated"] is not True:
                raise TrainingFailure(
                    "first-step LoRA B and kernel parameters must have finite gradients and update",
                    phase="optimizer_step",
                    step=step,
                    parameter=name,
                    value=dict(record),
                )
    if step >= 2:
        for name, record in records.items():
            if name.endswith(f".{ADAPTER_MODULE_NAME}.A") and (
                record["finite_grad"] is not True
                or not isinstance(record["relative_update"], float)
                or record["relative_update"] <= 0.0
            ):
                raise TrainingFailure(
                    "LoRA A must have a finite non-zero relative update after step one",
                    phase="optimizer_step",
                    step=step,
                    parameter=name,
                    value=dict(record),
                )
