"""OAL + LoRA payload construction and validation."""

from __future__ import annotations
from collections.abc import Mapping
from torch import Tensor, nn
from torch.optim import Optimizer
from .kernel_parameters import KernelParameterBank
from .checkpoint_context import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointKind,
    normalize_legacy_qwen_identity,
    CheckpointContext,
    _require_method,
    _normalise_metadata_mapping,
)
from .checkpoint_optimizer import _validate_optimizer_resume_state
from .checkpoint_state import (
    _snapshot_tensor_mapping,
    _validate_tensor_mapping,
    _require_payload_mapping,
    _payload_step,
    _payload_nll,
    _require_step,
    _require_finite_number,
    _validate_scheduler_state,
    _validate_rng_state,
)
from .checkpoint_targets import (
    _validate_runtime_objects,
    _lora_targets,
    _kernel_targets,
    _immutable_kernel_state_names,
)

_CORE_FIELDS = frozenset(
    (
        "schema_version",
        "kind",
        "method",
        "config_identity",
        "data_identity",
        "model_identity",
        "step",
        "validation_nll",
        "lora_state",
        "kernel_state",
    )
)
_RESUME_FIELDS = _CORE_FIELDS | frozenset(
    ("optimizer_state", "optimizer_bindings", "scheduler_state", "rng_state", "cursor")
)


def _build_core_payload(
    *,
    kind: CheckpointKind,
    model: nn.Module,
    kernel_bank: KernelParameterBank,
    context: CheckpointContext,
    step: int,
    validation_nll: float,
) -> dict[str, object]:
    _validate_runtime_objects(model=model, kernel_bank=kernel_bank, context=context)
    resolved_step = _require_step(step)
    resolved_nll = _require_finite_number(validation_nll, "validation_nll")
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": kind,
        "method": context.method,
        "config_identity": dict(context.config_identity),
        "data_identity": dict(context.data_identity),
        "model_identity": dict(context.model_identity),
        "step": resolved_step,
        "validation_nll": resolved_nll,
        "lora_state": _snapshot_tensor_mapping(_lora_targets(model), "lora_state"),
        "kernel_state": _snapshot_tensor_mapping(
            _kernel_targets(kernel_bank), "kernel_state"
        ),
    }


def _validate_core_payload(
    payload: object,
    *,
    expected_kind: CheckpointKind,
    model: nn.Module,
    kernel_bank: KernelParameterBank,
    context: CheckpointContext,
    payload_fields: frozenset[str] = _CORE_FIELDS,
    allow_historical_grouped_identity: bool = False,
) -> list[tuple[Tensor, Tensor]]:
    _validate_runtime_objects(model=model, kernel_bank=kernel_bank, context=context)
    record = _require_payload_mapping(
        payload, expected_fields=payload_fields, label="checkpoint"
    )
    _validate_checkpoint_identity_fields(
        record,
        expected_kind=expected_kind,
        model=model,
        context=context,
        allow_historical_grouped_identity=allow_historical_grouped_identity,
    )
    lora_actions = _validate_tensor_mapping(
        record["lora_state"], _lora_targets(model), "lora_state"
    )
    immutable_kernel_names = _immutable_kernel_state_names(kernel_bank)
    kernel_actions = _validate_tensor_mapping(
        record["kernel_state"],
        _kernel_targets(kernel_bank),
        "kernel_state",
        immutable_names=immutable_kernel_names,
    )
    return [*lora_actions, *kernel_actions]


def _validate_checkpoint_identity_fields(
    record: Mapping[str, object],
    *,
    expected_kind: CheckpointKind,
    model: nn.Module,
    context: CheckpointContext,
    allow_historical_grouped_identity: bool,
) -> None:
    if (
        record["schema_version"] != CHECKPOINT_SCHEMA_VERSION
        or type(record["schema_version"]) is not int
    ):
        raise ValueError(
            f"checkpoint schema_version must be {CHECKPOINT_SCHEMA_VERSION}"
        )
    if record["kind"] != expected_kind:
        raise ValueError(f"checkpoint kind must be {expected_kind!r}")
    if record["method"] != context.method:
        raise ValueError("checkpoint method does not match the supplied context")
    _require_method(record["method"])
    for field, expected in (
        ("config_identity", context.config_identity),
        ("data_identity", context.data_identity),
        ("model_identity", context.model_identity),
    ):
        actual = _normalise_metadata_mapping(record[field], field)
        expected = dict(expected)
        if field in {"config_identity", "model_identity"} and (
            "model_profile" in actual or "model_profile" in expected
        ):
            if "model_profile" not in actual:
                actual = normalize_legacy_qwen_identity(
                    actual, model=model, expected_identity=expected
                )
            if "model_profile" not in expected:
                expected = normalize_legacy_qwen_identity(
                    expected, model=model, expected_identity=actual
                )
        if field == "config_identity" and allow_historical_grouped_identity:
            if (
                actual.get("experiment_protocol_version")
                == "formal-lora-middle18-comparison-v1"
                and actual.get("method") == "grouped_quadratic"
                and (actual.get("tuning_mode") == "lora")
                and ("group_asset_sha256" not in actual)
                and ("group_asset_sha256" in expected)
            ):
                expected.pop("group_asset_sha256")
        if actual != expected:
            raise ValueError(f"checkpoint {field} does not match the supplied context")
    _payload_step(record)
    _payload_nll(record)


def _validate_resume_payload(
    payload: object,
    *,
    model: nn.Module,
    kernel_bank: KernelParameterBank,
    context: CheckpointContext,
    optimizer: Optimizer,
    scheduler: object | None,
) -> list[tuple[Tensor, Tensor]]:
    record = _require_payload_mapping(
        payload, expected_fields=_RESUME_FIELDS, label="resume checkpoint"
    )
    actions = _validate_core_payload(
        record,
        expected_kind="latest_resume",
        model=model,
        kernel_bank=kernel_bank,
        context=context,
        payload_fields=_RESUME_FIELDS,
    )
    _validate_optimizer_resume_state(
        optimizer_state=record["optimizer_state"],
        bindings=record["optimizer_bindings"],
        model=model,
        kernel_bank=kernel_bank,
        optimizer=optimizer,
    )
    _validate_scheduler_state(record["scheduler_state"], scheduler)
    _validate_rng_state(record["rng_state"])
    _normalise_metadata_mapping(record["cursor"], "cursor")
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer")
    return actions
