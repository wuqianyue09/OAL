"""The two-step full-model LoRA smoke gate and adapter topology checks."""

from __future__ import annotations
from collections.abc import Callable, Sequence
import time
from .config import METHOD_REGISTRY, PilotConfig
from .workflows.errors import OrchestrationError, _exception_record
from . import runtime_execution
from .smoke_common import (
    ProductionSmokeRuntime,
    _finalize_smoke_resources,
    _json_safe_smoke_scalar,
    _json_safe_smoke_update_audit,
    _require_smoke_gradients,
    _require_smoke_updates,
    _smoke_input_ids,
    _smoke_kernel_audit,
    _smoke_update_audit,
)


def _production_lora_full_model_step(
    runtime: ProductionSmokeRuntime, config: PilotConfig
) -> dict[str, object]:
    """Run the two-step full-length Qwen LoRA A/B production smoke gate."""
    import torch
    from .lora import ADAPTER_MODULE_NAME
    from .training import _backward_with_attention_validation, causal_next_token_nll

    bundle = runtime.bundle
    model = getattr(bundle, "model")
    kernel_bank = getattr(bundle, "kernel_bank")
    optimizer = getattr(bundle, "optimizer")
    scheduler = getattr(bundle, "scheduler")
    production_device = torch.device(getattr(bundle, "device"))
    geometry = config.geometry
    expected_lora_count = geometry.num_layers * len(config.lora.targets)
    expected_lora_targets = tuple(
        (
            f"model.layers.{layer_id}.self_attn.{target}"
            for layer_id in range(geometry.num_layers)
            for target in config.lora.targets
        )
    )
    base_report: dict[str, object] = {
        "sequence_length": config.sequence_length,
        "model_layers": geometry.num_layers,
        "optimizer_steps": 2,
        "lora": {
            "expected_a_count": expected_lora_count,
            "expected_b_count": expected_lora_count,
            "dtype": str(torch.float32),
        },
    }
    try:
        model_layers = getattr(getattr(model, "model", None), "layers", None)
        if model_layers is None or len(model_layers) != geometry.num_layers:
            raise OrchestrationError(
                f"full-model smoke requires exactly {geometry.num_layers} {config.resolved_model_profile} layers"
            )
        input_ids = _smoke_input_ids(
            assets=runtime.assets,
            sequence_length=config.sequence_length,
            device=getattr(bundle, "device"),
        )
        lora_a, lora_b = _collect_smoke_lora_pairs(
            model,
            adapter_module_name=ADAPTER_MODULE_NAME,
            expected_targets=expected_lora_targets,
        )
        kernel_parameters = {
            name: parameter
            for (name, parameter) in kernel_bank.named_parameters()
            if parameter.requires_grad
        }
        method_trains_kernel = METHOD_REGISTRY[config.method].trains_kernel_parameters
        if method_trains_kernel and (not kernel_parameters):
            raise OrchestrationError(
                f"{config.method} requires a trainable kernel parameter"
            )
        if not method_trains_kernel and kernel_parameters:
            raise OrchestrationError(
                f"{config.method} must not have a trainable kernel parameter"
            )
    except Exception as exc:
        return {
            **base_report,
            "state": "failed",
            "input_shape": [],
            "steps": [],
            "failed_step": None,
            "completed_optimizer_steps": 0,
            "failure": _exception_record(exc),
            "resources": {
                "elapsed_seconds": 0.0,
                "memory": {
                    "applicability": "not_applicable",
                    "device": str(production_device),
                    "reason": "two-step resource measurement did not start",
                },
            },
        }
    measures_cuda = production_device.type == "cuda"
    resource_initialization_phase = "timer_start"
    try:
        if measures_cuda:
            resource_initialization_phase = "cuda_synchronize"
            torch.cuda.synchronize(production_device)
            resource_initialization_phase = "cuda_reset_peak_memory_stats"
            torch.cuda.reset_peak_memory_stats(production_device)
        resource_initialization_phase = "timer_start"
        started_at = time.perf_counter()
    except Exception as exc:
        failure_record = {
            "phase": resource_initialization_phase,
            **_exception_record(exc),
        }
        return {
            **base_report,
            "state": "failed",
            "input_shape": list(input_ids.shape),
            "steps": [],
            "failed_step": {
                "step": None,
                "phase": resource_initialization_phase,
                "loss": None,
                "optimizer_step_completed": False,
                "scheduler_step_completed": False,
            },
            "completed_optimizer_steps": 0,
            "failure": failure_record,
            "resources": {
                "elapsed_seconds": 0.0,
                "memory": {
                    "applicability": "measurement_failed",
                    "device": str(production_device),
                    "peak_allocated_bytes": None,
                    "peak_reserved_bytes": None,
                    "failure": failure_record,
                },
            },
        }
    step_audits: list[dict[str, object]] = []
    current_step: dict[str, object] | None = None
    current_before: dict[str, object] = {}
    completed_optimizer_steps = 0
    failure: Exception | None = None
    resources: dict[str, object]
    try:
        for optimizer_step in (1, 2):
            current_before = {
                **{
                    f"lora_a:{name}": parameter.detach().clone()
                    for (name, parameter) in lora_a.items()
                },
                **{
                    f"lora_b:{name}": parameter.detach().clone()
                    for (name, parameter) in lora_b.items()
                },
                **{
                    f"kernel:{name}": parameter.detach().clone()
                    for (name, parameter) in kernel_parameters.items()
                },
            }
            current_step = {
                "step": optimizer_step,
                "phase": "zero_grad",
                "loss": None,
                "optimizer_step_completed": False,
                "scheduler_step_completed": False,
            }
            optimizer.zero_grad(set_to_none=True)
            current_step["phase"] = "train"
            model.train()
            kernel_bank.train()
            current_step["phase"] = "forward"
            logits = runtime_execution.model_logits(model, input_ids)
            loss = causal_next_token_nll(logits, input_ids.clone())
            current_step["loss"] = _json_safe_smoke_scalar(loss)
            current_step["phase"] = "loss_validation"
            runtime_execution.require_finite_tensor(
                loss, f"full-model causal loss at optimizer step {optimizer_step}"
            )
            current_step["phase"] = "backward"
            _backward_with_attention_validation(loss)
            current_step["phase"] = "gradient_validation"
            _require_smoke_gradients(
                lora_a, category="LoRA A", optimizer_step=optimizer_step
            )
            _require_smoke_gradients(
                lora_b, category="LoRA B", optimizer_step=optimizer_step
            )
            _require_smoke_gradients(
                kernel_parameters, category="kernel", optimizer_step=optimizer_step
            )
            current_step["phase"] = "optimizer_step"
            optimizer.step()
            completed_optimizer_steps += 1
            current_step["optimizer_step_completed"] = True
            current_step["phase"] = "scheduler_step"
            scheduler.step()
            current_step["scheduler_step_completed"] = True
            lora_a_audit = _smoke_update_audit(lora_a, current_before, prefix="lora_a:")
            lora_b_audit = _smoke_update_audit(lora_b, current_before, prefix="lora_b:")
            kernel_audit = _smoke_kernel_audit(
                kernel_parameters,
                current_before,
                prefix="kernel:",
                method=config.method,
                strict=True,
            )
            current_step.update(
                {
                    "lora_a": {
                        "gate": (
                            "observation_only"
                            if optimizer_step == 1
                            else "finite_gradient_and_update"
                        ),
                        "parameters": lora_a_audit,
                    },
                    "lora_b": {
                        "gate": "finite_gradient_and_update",
                        "parameters": lora_b_audit,
                    },
                    "kernel": kernel_audit,
                }
            )
            current_step["phase"] = "update_validation"
            _require_smoke_updates(
                lora_b_audit, category="LoRA B", optimizer_step=optimizer_step
            )
            if optimizer_step == 2:
                _require_smoke_updates(
                    lora_a_audit, category="LoRA A", optimizer_step=optimizer_step
                )
            if kernel_parameters:
                _require_smoke_updates(
                    kernel_audit["parameters"],
                    category="kernel",
                    optimizer_step=optimizer_step,
                )
            step_audits.append(
                {
                    "step": optimizer_step,
                    "loss": current_step["loss"],
                    "lora_a": current_step["lora_a"],
                    "lora_b": current_step["lora_b"],
                    "kernel": current_step["kernel"],
                }
            )
            current_step = None
    except Exception as exc:
        failure = exc
        if current_step is not None:
            audit_collection_failure: dict[str, object] = {}

            def collect_partial_audit(
                name: str, collector: Callable[[], object]
            ) -> None:
                if name in current_step:
                    return
                try:
                    current_step[name] = collector()
                except Exception as audit_error:
                    audit_collection_failure[name] = _exception_record(audit_error)

            collect_partial_audit(
                "lora_a",
                lambda: {
                    "gate": (
                        "observation_only"
                        if current_step["step"] == 1
                        else "finite_gradient_and_update"
                    ),
                    "parameters": _json_safe_smoke_update_audit(
                        lora_a, current_before, prefix="lora_a:"
                    ),
                },
            )
            collect_partial_audit(
                "lora_b",
                lambda: {
                    "gate": "finite_gradient_and_update",
                    "parameters": _json_safe_smoke_update_audit(
                        lora_b, current_before, prefix="lora_b:"
                    ),
                },
            )
            collect_partial_audit(
                "kernel",
                lambda: _smoke_kernel_audit(
                    kernel_parameters,
                    current_before,
                    prefix="kernel:",
                    method=config.method,
                    strict=False,
                ),
            )
            if audit_collection_failure:
                current_step["audit_collection_failure"] = audit_collection_failure
    finally:
        try:
            resources = _finalize_smoke_resources(
                production_device, measures_cuda=measures_cuda, started_at=started_at
            )
        except Exception as resource_error:
            if failure is None:
                failure = resource_error
            resources = {
                "elapsed_seconds": 0.0,
                "memory": {
                    "applicability": "measurement_failed",
                    "device": str(production_device),
                    "peak_allocated_bytes": None,
                    "peak_reserved_bytes": None,
                    "failure": _exception_record(resource_error),
                },
            }
    if failure is not None:
        return {
            **base_report,
            "state": "failed",
            "input_shape": list(input_ids.shape),
            "steps": step_audits,
            "failed_step": current_step,
            "completed_optimizer_steps": completed_optimizer_steps,
            "failure": _exception_record(failure),
            "resources": resources,
        }
    return {
        **base_report,
        "state": "passed",
        "input_shape": list(input_ids.shape),
        "steps": step_audits,
        "resources": resources,
    }


def _collect_smoke_lora_pairs(
    model: object, *, adapter_module_name: str, expected_targets: Sequence[str]
) -> tuple[dict[str, object], dict[str, object]]:
    """Collect the exact trainable FP32 A/B pair set for the two-step gate."""
    import torch

    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise OrchestrationError("full-model smoke model has no named parameters")
    parameters = dict(named_parameters())
    lora_a = {
        name: parameter
        for (name, parameter) in parameters.items()
        if name.endswith(f".{adapter_module_name}.A")
    }
    lora_b = {
        name: parameter
        for (name, parameter) in parameters.items()
        if name.endswith(f".{adapter_module_name}.B")
    }
    expected_bases = {f"{target}.{adapter_module_name}" for target in expected_targets}
    expected_names = {
        "LoRA A": {f"{base}.A" for base in expected_bases},
        "LoRA B": {f"{base}.B" for base in expected_bases},
    }
    expected_count = len(expected_bases)
    for category, selected in (("LoRA A", lora_a), ("LoRA B", lora_b)):
        if len(selected) != expected_count:
            raise OrchestrationError(
                f"full-model smoke requires one trainable {category} for every layer/target; expected {expected_count}, got {len(selected)}"
            )
        if set(selected) != expected_names[category]:
            missing = sorted(expected_names[category] - set(selected))
            unexpected = sorted(set(selected) - expected_names[category])
            raise OrchestrationError(
                f"full-model smoke LoRA parameters do not match the canonical 24-layer target topology; missing={missing[:1]}, unexpected={unexpected[:1]}"
            )
        for name, parameter in selected.items():
            if not isinstance(parameter, torch.nn.Parameter):
                raise OrchestrationError(
                    f"{category} parameter {name} is not a Parameter"
                )
            if parameter.dtype != torch.float32:
                raise OrchestrationError(
                    f"{category} parameter {name} must be FP32; got {parameter.dtype}"
                )
            if not parameter.requires_grad:
                raise OrchestrationError(
                    f"{category} parameter {name} is not trainable"
                )
    a_targets = {name.removesuffix(".A") for name in lora_a}
    b_targets = {name.removesuffix(".B") for name in lora_b}
    if a_targets != b_targets:
        raise OrchestrationError(
            "full-model smoke LoRA A/B parameter targets are not paired"
        )
    return (lora_a, lora_b)
