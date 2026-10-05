"""Live LoRA and kernel tensor ownership for checkpoint operations."""

from __future__ import annotations
from torch import Tensor, nn
from .kernel_parameters import KernelParameterBank
from .lora import LoRAAdapter
from .checkpoint_context import CheckpointContext


def _validate_runtime_objects(
    *, model: nn.Module, kernel_bank: KernelParameterBank, context: CheckpointContext
) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(kernel_bank, KernelParameterBank):
        raise TypeError("kernel_bank must be a KernelParameterBank")
    if not isinstance(context, CheckpointContext):
        raise TypeError("context must be a CheckpointContext")
    if kernel_bank.method != context.method:
        raise ValueError(
            f"kernel_bank method {kernel_bank.method!r} does not match context method {context.method!r}"
        )


def _lora_targets(model: nn.Module) -> dict[str, Tensor]:
    """Return exactly direct A/B parameters of registered real LoRA adapters."""
    targets: dict[str, Tensor] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, LoRAAdapter):
            continue
        if not module_name:
            raise ValueError("LoRAAdapter must be registered below the model root")
        direct_parameters = dict(module.named_parameters(recurse=False))
        if set(direct_parameters) != {"A", "B"}:
            raise ValueError(
                f"LoRA adapter {module_name} must have exactly A and B parameters"
            )
        for parameter_name in ("A", "B"):
            parameter = direct_parameters[parameter_name]
            if not isinstance(parameter, nn.Parameter):
                raise ValueError(
                    f"LoRA adapter {module_name}.{parameter_name} is not a parameter"
                )
            name = f"{module_name}.{parameter_name}"
            if name in targets:
                raise ValueError(f"duplicate LoRA adapter parameter name: {name}")
            targets[name] = parameter
    if not targets:
        raise ValueError("model has no registered LoRA A/B parameters to checkpoint")
    return dict(sorted(targets.items()))


def _kernel_targets(kernel_bank: KernelParameterBank) -> dict[str, Tensor]:
    targets: dict[str, Tensor] = {}
    for name, parameter in kernel_bank.named_parameters():
        targets[name] = parameter
    for name, buffer in kernel_bank.named_buffers():
        if name in targets:
            raise ValueError(f"kernel bank state name is ambiguous: {name}")
        targets[name] = buffer
    state_keys = set(kernel_bank.state_dict())
    if state_keys != set(targets):
        raise ValueError(
            "kernel bank state_dict contains undeclared parameter or buffer state"
        )
    return dict(sorted(targets.items()))


def _immutable_kernel_state_names(kernel_bank: KernelParameterBank) -> frozenset[str]:
    """Return the immutable OAL group/layout buffers matched before loading."""
    return frozenset((name for (name, _) in kernel_bank.named_buffers()))
