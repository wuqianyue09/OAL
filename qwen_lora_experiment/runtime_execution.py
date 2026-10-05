"""Shared production device, forward validation, and attention execution contracts."""

from __future__ import annotations
from collections.abc import Mapping
from .config import GROUPED_BASE_METHODS, MethodName, PilotConfig
from .workflows.errors import OrchestrationError

_GROUPED_ATTENTION_EXECUTION_FIELDS = frozenset(
    (
        "method",
        "method_identity",
        "execution",
        "callable",
        "experimental",
        "admission",
        "group_asset_mode",
        "group_asset_status",
        "group_asset_sequence_length",
        "replaced_layer_ids_zero_based",
    )
)
_SOFTMAX_ATTENTION_EXECUTION_FIELDS = frozenset(
    ("method", "execution", "callable", "experimental", "replaced_layer_ids_zero_based")
)


def _runtime_attention_execution(
    config: PilotConfig, runtime: object
) -> dict[str, object]:
    """Validate and copy the execution proven by the assembled model bundle."""
    evidence = getattr(runtime, "attention_execution", None)
    if not isinstance(evidence, Mapping):
        raise OrchestrationError("model runtime attention_execution must be a mapping")
    normalized = dict(evidence)
    if normalized.get("method") != config.method:
        raise OrchestrationError(
            "model runtime attention execution method does not match config"
        )
    if normalized.get("execution") != config.attention_execution:
        raise OrchestrationError(
            "model runtime attention execution does not match config"
        )
    callable_name = normalized.get("callable")
    if not isinstance(callable_name, str) or not callable_name:
        raise OrchestrationError(
            "model runtime attention execution must record a callable"
        )
    if type(normalized.get("experimental")) is not bool:
        raise OrchestrationError(
            "model runtime attention execution must record experimental"
        )
    if config.attention_backend == "hd_block_gemm":
        expected_fields = {
            "method",
            "method_identity",
            "execution",
            "callable",
            "experimental",
            "precision",
            "requested_options",
            "options",
            "logical_feature_width",
            "packed_pair_count",
            "physical_repeat_kv",
            "replaced_layer_ids_zero_based",
        }
        expected_callable = {
            "grouped_quadratic": "oal_attention.hd_block_gemm_adapters.run_prepared_grouped_hd_block_gemm"
        }[config.method]
        if set(normalized) != expected_fields:
            raise OrchestrationError(
                "HD attention execution evidence fields are incomplete"
            )
        actual_options = normalized.get("options")
        requested_options = config.resolved_hd_options.to_dict()
        if (
            callable_name != expected_callable
            or normalized["precision"] != config.resolved_hd_options.precision
            or normalized["requested_options"] != requested_options
            or (not isinstance(actual_options, Mapping))
            or (set(actual_options) != set(requested_options))
            or (actual_options.get("precision") != normalized["precision"])
            or any(
                (
                    actual_options[name] != value
                    for (name, value) in requested_options.items()
                    if name != "feature_wave_blocks"
                )
            )
            or (
                not (
                    isinstance(actual_options.get("feature_wave_blocks"), int)
                    and 1
                    <= actual_options["feature_wave_blocks"]
                    <= requested_options["feature_wave_blocks"]
                )
            )
            or (normalized["logical_feature_width"] != 2145)
            or (normalized["packed_pair_count"] != 2080)
            or (normalized["physical_repeat_kv"] is not False)
            or (
                normalized["replaced_layer_ids_zero_based"]
                != list(config.replacement_layer_ids)
            )
        ):
            raise OrchestrationError(
                "HD attention execution evidence does not match config"
            )
        return normalized
    if config.method == "grouped_quadratic":
        if set(normalized) != _GROUPED_ATTENTION_EXECUTION_FIELDS:
            raise OrchestrationError(
                "grouped execution evidence fields must exactly match the public-operator schema"
            )
        if callable_name != "oal_attention.oal_attention":
            raise OrchestrationError(
                "grouped execution evidence must record only oal_attention.oal_attention"
            )
        if (
            normalized["execution"] != "triton"
            or normalized["experimental"] is not False
        ):
            raise OrchestrationError(
                "grouped execution evidence must record public Triton execution"
            )
        if normalized.get("admission") != "public_fail_closed":
            raise OrchestrationError(
                "grouped execution evidence must record public fail-closed admission"
            )
        if normalized.get("replaced_layer_ids_zero_based") != list(
            config.replacement_layer_ids
        ):
            raise OrchestrationError(
                "grouped execution evidence replacement layers do not match the effective configuration"
            )
    return normalized


def _production_cuda_device() -> object:
    """Resolve the only supported production device at runtime, not import time."""
    import torch

    if not torch.cuda.is_available():
        raise OrchestrationError("pilot training requires an available CUDA device")
    return torch.device("cuda:0")


def _method_requires_oal_attention(method: MethodName) -> bool:
    """Operator-backed methods require the OAL package during preflight."""
    return method in {*()} or method in GROUPED_BASE_METHODS


def model_logits(model: object, input_ids: object) -> object:
    """Call a CausalLM without delegating loss/label semantics to Transformers."""
    from .attention.common import run_attention_validation_forward

    output = run_attention_validation_forward(
        model, input_ids=input_ids, use_cache=False, return_dict=True
    )
    logits = getattr(output, "logits", None)
    if logits is None:
        raise OrchestrationError("model output has no logits Tensor")
    return logits


def require_finite_tensor(value: object, name: str) -> None:
    """Fail a stage before success evidence can be written for bad numerics."""
    import torch

    if not isinstance(value, torch.Tensor):
        raise OrchestrationError(f"{name} is not a Tensor")
    if not bool(torch.isfinite(value).all()):
        raise OrchestrationError(f"{name} contains non-finite values")
