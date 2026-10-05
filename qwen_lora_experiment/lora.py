"""Explicit, hook-based LoRA injection for the supported pilot topologies.

The pilot deliberately avoids weight merging and parametrization.  Each target
``nn.Linear`` owns a registered :class:`LoRAAdapter`, while a forward hook adds
only the low-rank residual to the unchanged linear output.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from collections.abc import Iterable
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .backbones.spec import ModelGeometry, geometry_for_profile
from .config import DEFAULT_LORA_TARGETS, LoRAConfig

ADAPTER_MODULE_NAME = "_qwen_lora_adapter"
_HOOK_HANDLE_ATTRIBUTE = "_qwen_lora_hook_handle"
_BASE_TRAINABILITY_ATTRIBUTE = "_qwen_lora_base_trainability"


@dataclass(frozen=True)
class ParameterRecord:
    """A serialisable description of one trainable model parameter."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    requires_grad: bool


@dataclass(frozen=True)
class TrainableParameterReport:
    """The optimizer-safe set of parameters after LoRA injection."""

    trainable: tuple[ParameterRecord, ...]

    @property
    def trainable_names(self) -> tuple[str, ...]:
        return tuple((record.name for record in self.trainable))

    @property
    def total_trainable_parameters(self) -> int:
        return sum((math.prod(record.shape) for record in self.trainable))


class LoRAAdapter(nn.Module):
    """The FP32 low-rank residual ``(alpha / rank) * B(A(dropout(input)))``.

    ``A`` and ``B`` remain direct leaf parameters.  The surrounding linear
    module owns this adapter, so normal module device moves and ``state_dict``
    operations include the adapter without any hidden parameter references.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout: float,
        seed: int,
        *,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError("in_features must be positive")
        if out_features <= 0:
            raise ValueError("out_features must be positive")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.dropout = nn.Dropout(p=dropout)
        self.A = nn.Parameter(
            torch.empty((rank, in_features), dtype=torch.float32, device=device)
        )
        self.B = nn.Parameter(
            torch.zeros((out_features, rank), dtype=torch.float32, device=device)
        )
        initialization_generator = generator or _new_generator(self.A.device, seed)
        with torch.no_grad():
            nn.init.kaiming_uniform_(
                self.A, a=math.sqrt(5), generator=initialization_generator
            )

    def forward(
        self,
        input: Tensor,
        *,
        output_dtype: torch.dtype,
        output_device: torch.device | None = None,
    ) -> Tensor:
        """Return only the LoRA residual in the requested base-output format."""
        if not isinstance(input, Tensor):
            raise TypeError("LoRAAdapter input must be a torch.Tensor")
        computation_input = input.to(device=self.A.device, dtype=self.A.dtype)
        residual = F.linear(F.linear(self.dropout(computation_input), self.A), self.B)
        residual = residual * self.scale
        return residual.to(
            device=output_device if output_device is not None else input.device,
            dtype=output_dtype,
        )

    def _apply(self, fn: object, recurse: bool = True) -> LoRAAdapter:
        """Follow device migration while retaining FP32 optimizer parameters.

        ``Module.to(dtype=...)`` supplies a conversion callable that would
        normally downcast every child parameter.  Probing that callable gives
        the requested device without applying its dtype conversion to A/B.
        """
        if not callable(fn):
            raise TypeError("LoRAAdapter._apply requires a tensor conversion callable")
        if recurse:
            for child in self.children():
                child._apply(fn)
        for parameter_name, parameter in tuple(self._parameters.items()):
            if parameter is None:
                continue
            with torch.no_grad():
                if parameter.device.type == "meta":
                    converted = fn(parameter).to(dtype=torch.float32)
                else:
                    device_probe = torch.empty(
                        0, dtype=parameter.dtype, device=parameter.device
                    )
                    target_device = fn(device_probe).device
                    converted = parameter.to(device=target_device, dtype=torch.float32)
            original_grad = parameter.grad
            if converted.device == parameter.device:
                parameter.data = converted
                output_parameter = parameter
            else:
                output_parameter = nn.Parameter(
                    converted, requires_grad=parameter.requires_grad
                )
                self._parameters[parameter_name] = output_parameter
            if original_grad is not None:
                if original_grad.device.type == "meta":
                    converted_grad = fn(original_grad).to(dtype=torch.float32)
                else:
                    converted_grad = original_grad.to(
                        device=output_parameter.device, dtype=torch.float32
                    )
                output_parameter.grad = converted_grad
        for buffer_name, buffer in self._buffers.items():
            if buffer is not None:
                self._buffers[buffer_name] = fn(buffer)
        return self


def inject_lora_adapters(
    model: nn.Module,
    config: LoRAConfig,
    seed: int,
    *,
    geometry: ModelGeometry | None = None,
) -> list[str]:
    """Freeze a base model and attach LoRA adapters to every layer projection.

    Only ``model.layers.<0..23>.self_attn.{q,k,v,o}_proj`` is accepted.  Target
    discovery completes before mutating the model, so malformed model shapes do
    not leave a partially injected module tree behind.  An optimizer created
    before injection cannot acquire the new adapter parameters and is therefore
    unsupported; create a fresh optimizer after this function returns.
    """
    config.validate()
    if hasattr(model, _BASE_TRAINABILITY_ATTRIBUTE):
        raise RuntimeError(
            "LoRA adapters are already injected; remove them before reinjecting"
        )
    resolved_geometry = geometry or geometry_for_profile("qwen2_5_0_5b")
    targets = _validate_injection_targets(model, config, resolved_geometry)
    base_trainability = {
        name: parameter.requires_grad for (name, parameter) in model.named_parameters()
    }
    setattr(model, _BASE_TRAINABILITY_ATTRIBUTE, base_trainability)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    generators: dict[str, torch.Generator] = {}
    for name, linear in targets:
        device_key = str(linear.weight.device)
        generator = generators.get(device_key)
        if generator is None:
            generator = _new_generator(linear.weight.device, seed)
            generators[device_key] = generator
        adapter = LoRAAdapter(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            seed=seed,
            device=linear.weight.device,
            generator=generator,
        )
        adapter._target_name = name
        linear.add_module(ADAPTER_MODULE_NAME, adapter)
        setattr(
            linear, _HOOK_HANDLE_ATTRIBUTE, linear.register_forward_hook(_lora_hook)
        )
    return [name for (name, _) in targets]


def remove_lora_adapters(
    model: nn.Module, *, geometry: ModelGeometry | None = None
) -> list[str]:
    """Remove adapters/hooks and restore the exact original base trainability.

    Every stored hook is removed even if its adapter has been externally
    corrupted.  Recreate any optimizer after removal: an optimizer created
    while adapters were present retains removed parameter references and is not
    safe to reuse.
    """
    removed: list[str] = []
    for name in _expected_target_names(geometry):
        try:
            module = model.get_submodule(name)
        except AttributeError:
            continue
        if not isinstance(module, nn.Linear):
            continue
        handle = getattr(module, _HOOK_HANDLE_ATTRIBUTE, None)
        has_adapter = hasattr(module, ADAPTER_MODULE_NAME)
        if handle is not None:
            handle.remove()
            delattr(module, _HOOK_HANDLE_ATTRIBUTE)
        if has_adapter:
            delattr(module, ADAPTER_MODULE_NAME)
        if handle is not None or has_adapter:
            removed.append(name)
    base_trainability = getattr(model, _BASE_TRAINABILITY_ATTRIBUTE, None)
    if base_trainability is not None:
        named_parameters = dict(model.named_parameters())
        for name, requires_grad in base_trainability.items():
            parameter = named_parameters.get(name)
            if parameter is None:
                raise RuntimeError(
                    f"cannot restore LoRA base parameter trainability; parameter is missing: {name}"
                )
            parameter.requires_grad_(requires_grad)
            parameter.grad = None
        delattr(model, _BASE_TRAINABILITY_ATTRIBUTE)
    return removed


def assert_parameter_ownership(
    model: nn.Module,
    method: str,
    *,
    kernel_parameter_names: Iterable[str] = (),
    geometry: ModelGeometry | None = None,
) -> TrainableParameterReport:
    """Require that only LoRA and explicitly declared kernel parameters train.

    ``kernel_parameter_names`` are full names from ``model.named_parameters()``;
    callers must opt into every non-LoRA trainable parameter.  ``method`` is
    accepted here to keep the runner's per-method audit call explicit, while
    method-specific kernel construction remains outside this generic module.
    """
    del method
    named_parameters = dict(model.named_parameters())
    declared_kernels = tuple(kernel_parameter_names)
    if len(set(declared_kernels)) != len(declared_kernels):
        raise ValueError("kernel_parameter_names must not contain duplicates")
    for name in declared_kernels:
        if name not in named_parameters:
            raise ValueError(f"declared kernel parameter is not registered: {name}")
    _assert_registered_adapter_types(model, geometry)
    expected_adapter_parameters = _expected_adapter_parameter_names(geometry)
    adapter_parameters = _registered_adapter_parameter_names(named_parameters)
    missing_adapters = expected_adapter_parameters - adapter_parameters
    if missing_adapters:
        raise RuntimeError(
            f"missing registered LoRA adapter parameter: {sorted(missing_adapters)[0]}"
        )
    unexpected_adapters = adapter_parameters - expected_adapter_parameters
    if unexpected_adapters:
        raise RuntimeError(
            f"unexpected registered LoRA adapter parameter: {sorted(unexpected_adapters)[0]}"
        )
    expected_trainable = adapter_parameters | set(declared_kernels)
    records: list[ParameterRecord] = []
    for name, parameter in named_parameters.items():
        if name in adapter_parameters:
            _assert_fp32_adapter_parameter(name, parameter)
            if not parameter.requires_grad:
                raise RuntimeError(f"LoRA adapter parameter must train: {name}")
        elif name in declared_kernels:
            if not parameter.requires_grad:
                raise RuntimeError(f"declared kernel parameter must train: {name}")
        elif parameter.requires_grad:
            raise RuntimeError(
                f"parameter ownership audit found unexpected trainable base parameter: {name}"
            )
        if name in expected_trainable:
            records.append(_parameter_record(name, parameter))
    return TrainableParameterReport(
        trainable=tuple(sorted(records, key=lambda item: item.name))
    )


def assert_all_on_device(
    model: nn.Module,
    *,
    device: torch.device,
    input_tensor: Tensor | None = None,
    kernel_parameter_names: Iterable[str] = (),
) -> None:
    """Fail closed when runtime inputs, parameters, or adapter dtypes drift."""
    if input_tensor is not None and input_tensor.device != device:
        _raise_device_dtype_error(
            "input", expected_device=device, expected_dtype=None, actual=input_tensor
        )
    named_parameters = dict(model.named_parameters())
    declared_kernels = tuple(kernel_parameter_names)
    for name in declared_kernels:
        if name not in named_parameters:
            raise ValueError(f"declared kernel parameter is not registered: {name}")
    adapter_parameters = _registered_adapter_parameter_names(named_parameters)
    for name, parameter in named_parameters.items():
        expected_dtype = (
            torch.float32
            if name in adapter_parameters or name in declared_kernels
            else None
        )
        if parameter.device != device or (
            expected_dtype is not None and parameter.dtype != expected_dtype
        ):
            _raise_device_dtype_error(
                name,
                expected_device=device,
                expected_dtype=expected_dtype,
                actual=parameter,
            )


def _lora_hook(module: nn.Module, inputs: tuple[object, ...], output: object) -> Tensor:
    if not isinstance(module, nn.Linear):
        raise RuntimeError("LoRA hook was attached to a non-Linear module")
    if not inputs or not isinstance(inputs[0], Tensor):
        raise TypeError("LoRA hook requires a tensor as the first Linear input")
    if not isinstance(output, Tensor):
        raise TypeError("LoRA hook requires a tensor Linear output")
    adapter = getattr(module, ADAPTER_MODULE_NAME, None)
    if not isinstance(adapter, LoRAAdapter):
        raise RuntimeError("LoRA hook is missing its registered adapter")
    input_tensor = inputs[0]
    _assert_forward_devices(module, adapter, input_tensor, output)
    return output + adapter(
        input_tensor, output_dtype=output.dtype, output_device=output.device
    )


def _validate_injection_targets(
    model: nn.Module, config: LoRAConfig, geometry: ModelGeometry
) -> list[tuple[str, nn.Linear]]:
    if tuple(config.targets) != DEFAULT_LORA_TARGETS:
        raise ValueError(
            "LoRA injection targets must be exactly ('q_proj', 'k_proj', 'v_proj', 'o_proj')"
        )
    targets: list[tuple[str, nn.Linear]] = []
    for name in _expected_target_names(geometry):
        try:
            module = model.get_submodule(name)
        except AttributeError as exc:
            raise ValueError(f"missing required LoRA target: {name}") from exc
        if not isinstance(module, nn.Linear):
            raise TypeError(f"LoRA target must be nn.Linear: {name}")
        if hasattr(module, ADAPTER_MODULE_NAME) or hasattr(
            module, _HOOK_HANDLE_ATTRIBUTE
        ):
            raise RuntimeError(f"LoRA adapters already injected at {name}")
        targets.append((name, module))
    return targets


def _expected_target_names(geometry: ModelGeometry | None = None) -> tuple[str, ...]:
    resolved_geometry = geometry or geometry_for_profile("qwen2_5_0_5b")
    return tuple(
        (
            f"model.layers.{layer}.self_attn.{projection}"
            for layer in range(resolved_geometry.num_layers)
            for projection in DEFAULT_LORA_TARGETS
        )
    )


def _new_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _registered_adapter_parameter_names(
    named_parameters: dict[str, nn.Parameter],
) -> set[str]:
    marker = f".{ADAPTER_MODULE_NAME}."
    return {name for name in named_parameters if marker in name}


def _expected_adapter_parameter_names(
    geometry: ModelGeometry | None = None,
) -> set[str]:
    return {
        f"{target_name}.{ADAPTER_MODULE_NAME}.{parameter_name}"
        for target_name in _expected_target_names(geometry)
        for parameter_name in ("A", "B")
    }


def _assert_registered_adapter_types(
    model: nn.Module, geometry: ModelGeometry | None = None
) -> None:
    for target_name in _expected_target_names(geometry):
        try:
            target = model.get_submodule(target_name)
        except AttributeError as exc:
            raise RuntimeError(f"missing LoRA target: {target_name}") from exc
        adapter = target._modules.get(ADAPTER_MODULE_NAME)
        if not isinstance(adapter, LoRAAdapter):
            raise RuntimeError(f"registered adapter must be LoRAAdapter: {target_name}")


def _assert_fp32_adapter_parameter(name: str, parameter: nn.Parameter) -> None:
    if parameter.dtype is not torch.float32:
        _raise_device_dtype_error(
            name,
            expected_device=parameter.device,
            expected_dtype=torch.float32,
            actual=parameter,
        )
    if not parameter.is_leaf:
        raise RuntimeError(f"LoRA adapter parameter must be a leaf: {name}")


def _assert_forward_devices(
    linear: nn.Linear, adapter: LoRAAdapter, input_tensor: Tensor, output: Tensor
) -> None:
    target_name = getattr(adapter, "_target_name", "LoRAAdapter")
    expected_device = linear.weight.device
    checks = (
        (f"{target_name}.input", input_tensor, None),
        (f"{target_name}.weight", linear.weight, None),
        (f"{target_name}.{ADAPTER_MODULE_NAME}.A", adapter.A, torch.float32),
        (f"{target_name}.{ADAPTER_MODULE_NAME}.B", adapter.B, torch.float32),
        (f"{target_name}.output", output, None),
    )
    if linear.bias is not None:
        checks += ((f"{target_name}.bias", linear.bias, None),)
    for name, tensor, expected_dtype in checks:
        if tensor.device != expected_device or (
            expected_dtype is not None and tensor.dtype != expected_dtype
        ):
            _raise_device_dtype_error(
                name,
                expected_device=expected_device,
                expected_dtype=expected_dtype,
                actual=tensor,
            )


def _parameter_record(name: str, parameter: nn.Parameter) -> ParameterRecord:
    return ParameterRecord(
        name=name,
        shape=tuple(parameter.shape),
        dtype=parameter.dtype,
        device=parameter.device,
        requires_grad=parameter.requires_grad,
    )


def _raise_device_dtype_error(
    name: str,
    *,
    expected_device: torch.device,
    expected_dtype: torch.dtype | None,
    actual: Tensor,
) -> None:
    expected = f"device={expected_device}"
    if expected_dtype is not None:
        expected += f", dtype={expected_dtype}"
    raise RuntimeError(
        f"LoRA device/dtype audit failed for {name}: expected {expected}; got device={actual.device}, dtype={actual.dtype}"
    )
