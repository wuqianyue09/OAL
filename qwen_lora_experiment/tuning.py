"""Dependency-light policy for OAL + LoRA training."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from .config import METHOD_REGISTRY, MethodName, TuningMode

TrainingRole = Literal["lora_training"]


@dataclass(frozen=True)
class TuningPolicy:
    """Immutable behavior selected before model construction."""

    tuning_mode: TuningMode
    training_role: TrainingRole
    inject_lora: bool
    train_kernel: bool
    requires_optimizer: bool


def resolve_tuning_policy(
    tuning_mode: TuningMode | str, method: MethodName | str
) -> TuningPolicy:
    if tuning_mode != "lora":
        raise ValueError("tuning_mode must be 'lora'")
    if method not in METHOD_REGISTRY:
        raise ValueError(
            f"method must be one of {tuple(METHOD_REGISTRY)}; got {method!r}"
        )
    return TuningPolicy(
        tuning_mode="lora",
        training_role="lora_training",
        inject_lora=True,
        train_kernel=True,
        requires_optimizer=True,
    )


__all__ = ["TrainingRole", "TuningPolicy", "resolve_tuning_policy"]
