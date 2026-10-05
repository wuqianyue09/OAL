"""Optimizer parameter bindings and saved-state validation."""

from __future__ import annotations
from collections.abc import Mapping
import math
from torch import Tensor, nn
from torch.optim import Optimizer
from .kernel_parameters import KernelParameterBank
from .checkpoint_state import (
    _tensor_sha256,
    _checkpoint_tree_sha256,
    _checkpoint_tree_key,
    _require_payload_mapping,
    _validate_checkpoint_tree,
)
from .checkpoint_targets import _lora_targets

_OPTIMIZER_BINDING_SCHEMA_VERSION = 1
_OPTIMIZER_BINDING_FIELDS = frozenset(
    ("schema_version", "param_groups", "parameter_state_indices", "state_records")
)
_OPTIMIZER_GROUP_BINDING_FIELDS = frozenset(
    ("parameter_names", "state_indices", "hyperparameters", "hyperparameters_sha256")
)
_OPTIMIZER_STATE_RECORD_FIELDS = frozenset(
    ("owner_parameter_name", "tensor_specs", "signature_sha256")
)
_OPTIMIZER_TENSOR_SPEC_FIELDS = frozenset(("path", "shape", "dtype", "sha256"))


def _optimizer_allowed_parameters(
    *, model: nn.Module, kernel_bank: KernelParameterBank
) -> dict[int, str]:
    """Return the one-to-one identity/name allow-list for optimizer ownership."""
    allowed: dict[int, str] = {
        id(parameter): name for (name, parameter) in _lora_targets(model).items()
    }
    for name, parameter in kernel_bank.named_parameters():
        if id(parameter) in allowed:
            raise ValueError(f"checkpoint allow-list has duplicate parameter: {name}")
        allowed[id(parameter)] = f"kernel_state.{name}"
    return allowed


def _validate_optimizer_ownership(
    *, model: nn.Module, kernel_bank: KernelParameterBank, optimizer: Optimizer
) -> dict[int, str]:
    """Require the optimizer to own exactly the checkpointable trainable state.

    Optimizer ``param_groups`` contain parameter references even when no moment
    has been created yet.  Checking them here prevents a frozen base parameter
    from being smuggled into a resume file through a future accidental gradient.
    """
    owners = _checkpointable_optimizer_parameters(model=model, kernel_bank=kernel_bank)
    allowed = _validate_optimizer_ownership_for_parameters(
        checkpointable_parameters=owners, optimizer=optimizer
    )
    legacy_allowed = _optimizer_allowed_parameters(model=model, kernel_bank=kernel_bank)
    if allowed != legacy_allowed:
        raise ValueError(
            "checkpoint optimizer allow-list changed from the LoRA contract"
        )
    return allowed


def _validate_optimizer_ownership_for_parameters(
    *, checkpointable_parameters: Mapping[str, nn.Parameter], optimizer: Optimizer
) -> dict[int, str]:
    """Require exact ownership of one explicit name-to-Parameter allow-list."""
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer")
    if not checkpointable_parameters:
        raise ValueError("checkpoint optimizer allow-list must not be empty")
    allowed: dict[int, str] = {}
    for name, parameter in checkpointable_parameters.items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                "checkpoint optimizer parameter names must be non-empty strings"
            )
        if not isinstance(parameter, nn.Parameter):
            raise ValueError(
                f"checkpointable optimizer tensor is not a parameter: {name}"
            )
        if id(parameter) in allowed:
            raise ValueError(f"checkpoint allow-list has duplicate parameter: {name}")
        allowed[id(parameter)] = name
    actual: dict[int, int] = {}
    outside: list[str] = []
    for group_index, group in enumerate(optimizer.param_groups):
        parameters = group.get("params")
        if not isinstance(parameters, list):
            raise ValueError(
                f"optimizer param_groups[{group_index}].params must be a list"
            )
        for parameter_index, parameter in enumerate(parameters):
            if not isinstance(parameter, nn.Parameter):
                raise ValueError(
                    f"optimizer param_groups[{group_index}].params[{parameter_index}] is not a Parameter"
                )
            parameter_id = id(parameter)
            if parameter_id not in allowed:
                outside.append(f"group {group_index} parameter {parameter_index}")
            actual[parameter_id] = actual.get(parameter_id, 0) + 1
    if outside:
        raise ValueError(
            "optimizer contains parameter(s) outside checkpoint allow-list: "
            + ", ".join(outside)
        )
    duplicates = [
        allowed[parameter_id] for (parameter_id, count) in actual.items() if count != 1
    ]
    missing = [
        name for (parameter_id, name) in allowed.items() if parameter_id not in actual
    ]
    if duplicates:
        raise ValueError(
            "optimizer contains duplicate checkpoint allow-list parameter(s): "
            + ", ".join(sorted(duplicates))
        )
    if missing:
        raise ValueError(
            "optimizer is missing checkpoint allow-list parameter(s): "
            + ", ".join(sorted(missing))
        )
    return allowed


def _checkpointable_optimizer_parameters(
    *, model: nn.Module, kernel_bank: KernelParameterBank
) -> dict[str, nn.Parameter]:
    """Return names and objects for exactly the optimizer-eligible tensors."""
    parameters: dict[str, nn.Parameter] = {}
    for name, parameter in _lora_targets(model).items():
        if not isinstance(parameter, nn.Parameter):
            raise ValueError(f"checkpointable LoRA tensor is not a parameter: {name}")
        parameters[name] = parameter
    for name, parameter in kernel_bank.named_parameters():
        full_name = f"kernel_state.{name}"
        if full_name in parameters:
            raise ValueError(
                f"checkpointable optimizer parameter name is ambiguous: {full_name}"
            )
        parameters[full_name] = parameter
    return parameters


def _build_optimizer_bindings(
    *,
    model: nn.Module,
    kernel_bank: KernelParameterBank,
    optimizer: Optimizer,
    optimizer_state: dict[str, object],
) -> dict[str, object]:
    """Bind saved optimizer state indexes to immutable checkpoint parameter names."""
    owners = _checkpointable_optimizer_parameters(model=model, kernel_bank=kernel_bank)
    _validate_optimizer_ownership(
        model=model, kernel_bank=kernel_bank, optimizer=optimizer
    )
    return _build_optimizer_bindings_for_parameters(
        checkpointable_parameters=owners,
        optimizer=optimizer,
        optimizer_state=optimizer_state,
    )


def _build_optimizer_bindings_for_parameters(
    *,
    checkpointable_parameters: Mapping[str, nn.Parameter],
    optimizer: Optimizer,
    optimizer_state: dict[str, object],
) -> dict[str, object]:
    """Bind optimizer indexes to one explicit checkpoint Parameter allow-list."""
    owners = dict(checkpointable_parameters)
    allowed = _validate_optimizer_ownership_for_parameters(
        checkpointable_parameters=owners, optimizer=optimizer
    )
    if set(owners) != set(allowed.values()):
        raise ValueError(
            "optimizer allow-list names do not match checkpointable parameters"
        )
    state_record = _require_payload_mapping(
        optimizer_state,
        expected_fields=frozenset(("state", "param_groups")),
        label="optimizer_state",
    )
    serialized_groups = state_record["param_groups"]
    if not isinstance(serialized_groups, list) or len(serialized_groups) != len(
        optimizer.param_groups
    ):
        raise ValueError(
            "optimizer_state.param_groups must match the live optimizer group count"
        )
    parameter_state_indices: dict[str, int] = {}
    group_bindings: list[dict[str, object]] = []
    index_owners: dict[int, str] = {}
    for group_index, (live_group, serialized_group) in enumerate(
        zip(optimizer.param_groups, serialized_groups, strict=True)
    ):
        if not isinstance(serialized_group, Mapping):
            raise ValueError(
                f"optimizer_state.param_groups[{group_index}] must be a mapping"
            )
        serialized_indices = serialized_group.get("params")
        live_parameters = live_group.get("params")
        if not isinstance(serialized_indices, list) or not isinstance(
            live_parameters, list
        ):
            raise ValueError(
                f"optimizer param group {group_index} must contain a params list"
            )
        if len(serialized_indices) != len(live_parameters):
            raise ValueError(
                f"optimizer param group {group_index} has inconsistent parameter counts"
            )
        names: list[str] = []
        indices: list[int] = []
        for parameter_index, (parameter, state_index) in enumerate(
            zip(live_parameters, serialized_indices, strict=True)
        ):
            if not isinstance(parameter, nn.Parameter):
                raise ValueError(
                    f"optimizer param_groups[{group_index}].params[{parameter_index}] is not a Parameter"
                )
            if type(state_index) is not int or state_index < 0:
                raise ValueError(
                    f"optimizer_state.param_groups[{group_index}].params[{parameter_index}] must be a non-negative integer state index"
                )
            name = allowed.get(id(parameter))
            if name is None:
                raise ValueError(
                    "optimizer contains parameter outside checkpoint allow-list"
                )
            if state_index in index_owners:
                raise ValueError(
                    f"optimizer state index {state_index} is bound to multiple parameters"
                )
            index_owners[state_index] = name
            parameter_state_indices[name] = state_index
            names.append(name)
            indices.append(state_index)
        hyperparameters = _canonical_optimizer_group_hyperparameters(
            serialized_group, f"optimizer_state.param_groups[{group_index}]"
        )
        group_bindings.append(
            {
                "parameter_names": names,
                "state_indices": indices,
                "hyperparameters": hyperparameters,
                "hyperparameters_sha256": _checkpoint_tree_sha256(hyperparameters),
            }
        )
    if set(parameter_state_indices) != set(owners):
        raise ValueError(
            "optimizer state-index bindings do not cover the checkpoint allow-list"
        )
    raw_states = state_record["state"]
    if not isinstance(raw_states, Mapping):
        raise ValueError("optimizer_state.state must be a mapping")
    state_indices = _validate_optimizer_state_indices(raw_states, index_owners)
    state_records: dict[str, object] = {}
    for state_index in state_indices:
        owner_name = index_owners[state_index]
        state_records[str(state_index)] = _build_optimizer_state_record(
            raw_states[state_index],
            owner=owners[owner_name],
            owner_name=owner_name,
            field=f"optimizer_state.state[{state_index}]",
        )
    return {
        "schema_version": _OPTIMIZER_BINDING_SCHEMA_VERSION,
        "param_groups": group_bindings,
        "parameter_state_indices": dict(sorted(parameter_state_indices.items())),
        "state_records": dict(
            sorted(state_records.items(), key=lambda item: int(item[0]))
        ),
    }


def _validate_optimizer_resume_state(
    *,
    optimizer_state: object,
    bindings: object,
    model: nn.Module,
    kernel_bank: KernelParameterBank,
    optimizer: Optimizer,
) -> None:
    """Reject any resume optimizer state not bound to this exact parameter layout.

    This validation intentionally runs before core adapter/kernel tensors are
    copied.  ``Optimizer.load_state_dict`` otherwise accepts matching shapes
    and may cast a malicious or corrupted float64 moment silently.
    """
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer")
    owners = _checkpointable_optimizer_parameters(model=model, kernel_bank=kernel_bank)
    allowed = _validate_optimizer_ownership(
        model=model, kernel_bank=kernel_bank, optimizer=optimizer
    )
    if set(owners) != set(allowed.values()):
        raise ValueError(
            "optimizer allow-list names do not match checkpointable parameters"
        )
    _validate_optimizer_resume_state_for_parameters(
        optimizer_state=optimizer_state,
        bindings=bindings,
        checkpointable_parameters=owners,
        optimizer=optimizer,
    )


def _validate_optimizer_resume_state_for_parameters(
    *,
    optimizer_state: object,
    bindings: object,
    checkpointable_parameters: Mapping[str, nn.Parameter],
    optimizer: Optimizer,
) -> None:
    """Validate resume state against one explicit Parameter allow-list."""
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer")
    owners = dict(checkpointable_parameters)
    allowed = _validate_optimizer_ownership_for_parameters(
        checkpointable_parameters=owners, optimizer=optimizer
    )
    if set(owners) != set(allowed.values()):
        raise ValueError(
            "optimizer allow-list names do not match checkpointable parameters"
        )
    state_record = _require_payload_mapping(
        optimizer_state,
        expected_fields=frozenset(("state", "param_groups")),
        label="optimizer_state",
    )
    binding_record = _require_payload_mapping(
        bindings, expected_fields=_OPTIMIZER_BINDING_FIELDS, label="optimizer_bindings"
    )
    if (
        type(binding_record["schema_version"]) is not int
        or binding_record["schema_version"] != _OPTIMIZER_BINDING_SCHEMA_VERSION
    ):
        raise ValueError(
            f"optimizer_bindings schema_version must be {_OPTIMIZER_BINDING_SCHEMA_VERSION}"
        )
    index_owners = _validate_optimizer_binding_groups(
        binding_record=binding_record,
        optimizer_state=state_record,
        optimizer=optimizer,
        allowed=allowed,
    )
    raw_states = state_record["state"]
    assert isinstance(raw_states, Mapping)
    state_indices = _validate_optimizer_state_indices(raw_states, index_owners)
    raw_state_records = binding_record["state_records"]
    if not isinstance(raw_state_records, Mapping):
        raise ValueError("optimizer_bindings.state_records must be a mapping")
    expected_record_keys = {str(index) for index in state_indices}
    actual_record_keys = set(raw_state_records)
    if actual_record_keys != expected_record_keys or not all(
        (isinstance(key, str) for key in actual_record_keys)
    ):
        raise ValueError(
            "optimizer_bindings.state_records must exactly match optimizer_state.state"
        )
    for state_index in state_indices:
        owner_name = index_owners[state_index]
        _validate_optimizer_state_record(
            raw_states[state_index],
            record=raw_state_records[str(state_index)],
            owner=owners[owner_name],
            owner_name=owner_name,
            field=f"optimizer_state.state[{state_index}]",
        )


def _validate_optimizer_binding_groups(
    *,
    binding_record: Mapping[str, object],
    optimizer_state: Mapping[str, object],
    optimizer: Optimizer,
    allowed: Mapping[int, str],
) -> dict[int, str]:
    raw_groups = binding_record["param_groups"]
    serialized_groups = optimizer_state["param_groups"]
    if not isinstance(raw_groups, list) or not isinstance(serialized_groups, list):
        raise ValueError(
            "optimizer bindings and optimizer_state must contain param_groups lists"
        )
    if len(raw_groups) != len(optimizer.param_groups) or len(serialized_groups) != len(
        optimizer.param_groups
    ):
        raise ValueError(
            "optimizer param-group count does not match the live optimizer"
        )
    raw_index_mapping = binding_record["parameter_state_indices"]
    if not isinstance(raw_index_mapping, Mapping):
        raise ValueError("optimizer_bindings.parameter_state_indices must be a mapping")
    expected_names = set(allowed.values())
    if set(raw_index_mapping) != expected_names or not all(
        (isinstance(name, str) for name in raw_index_mapping)
    ):
        raise ValueError(
            "optimizer_bindings.parameter_state_indices must exactly match the checkpoint allow-list"
        )
    name_indices: dict[str, int] = {}
    for name, state_index in raw_index_mapping.items():
        if type(state_index) is not int or state_index < 0:
            raise ValueError(
                f"optimizer_bindings state index for {name!r} must be a non-negative integer"
            )
        name_indices[name] = state_index
    if len(set(name_indices.values())) != len(name_indices):
        raise ValueError(
            "optimizer_bindings assigns one state index to multiple parameter names"
        )
    index_owners: dict[int, str] = {}
    for group_index, (raw_group, serialized_group, live_group) in enumerate(
        zip(raw_groups, serialized_groups, optimizer.param_groups, strict=True)
    ):
        group_binding = _require_payload_mapping(
            raw_group,
            expected_fields=_OPTIMIZER_GROUP_BINDING_FIELDS,
            label=f"optimizer_bindings.param_groups[{group_index}]",
        )
        names = group_binding["parameter_names"]
        state_indices = group_binding["state_indices"]
        serialized_indices = (
            serialized_group.get("params")
            if isinstance(serialized_group, Mapping)
            else None
        )
        live_parameters = live_group.get("params")
        if not isinstance(names, list) or not isinstance(state_indices, list):
            raise ValueError(
                f"optimizer_bindings.param_groups[{group_index}] must contain lists"
            )
        if not isinstance(serialized_indices, list) or not isinstance(
            live_parameters, list
        ):
            raise ValueError(
                f"optimizer_state.param_groups[{group_index}].params must be a list"
            )
        if (
            not len(names)
            == len(state_indices)
            == len(serialized_indices)
            == len(live_parameters)
        ):
            raise ValueError(
                f"optimizer param group {group_index} has inconsistent parameter counts"
            )
        for parameter_index, (
            name,
            state_index,
            serialized_index,
            parameter,
        ) in enumerate(
            zip(names, state_indices, serialized_indices, live_parameters, strict=True)
        ):
            if not isinstance(name, str) or name != allowed.get(id(parameter)):
                raise ValueError(
                    f"optimizer_bindings owner parameter name does not match live optimizer at group {group_index} parameter {parameter_index}"
                )
            if (
                type(state_index) is not int
                or state_index < 0
                or state_index != serialized_index
                or (state_index != name_indices[name])
            ):
                raise ValueError(
                    "optimizer_bindings state index does not match optimizer_state param-group structure"
                )
            if state_index in index_owners:
                raise ValueError(
                    f"optimizer state index {state_index} is bound to multiple parameters"
                )
            index_owners[state_index] = name
        _validate_optimizer_group_hyperparameters(
            serialized_group=serialized_group,
            group_binding=group_binding,
            group_index=group_index,
        )
    if set(index_owners.values()) != set(name_indices):
        raise ValueError(
            "optimizer bindings do not cover every checkpointable parameter"
        )
    return index_owners


def _validate_optimizer_state_indices(
    raw_states: Mapping[object, object], index_owners: Mapping[int, str]
) -> list[int]:
    state_indices: list[int] = []
    for state_index, state_value in raw_states.items():
        if type(state_index) is not int or state_index < 0:
            raise ValueError(
                "optimizer_state.state keys must be non-negative integer indexes"
            )
        if state_index not in index_owners:
            raise ValueError(f"optimizer_state has unknown state index {state_index}")
        if not isinstance(state_value, Mapping):
            raise ValueError(f"optimizer_state.state[{state_index}] must be a mapping")
        state_indices.append(state_index)
    return sorted(state_indices)


def _build_optimizer_state_record(
    value: object, *, owner: nn.Parameter, owner_name: str, field: str
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    _validate_checkpoint_tree(value, field)
    tensor_entries = _optimizer_tensor_entries(value)
    _validate_optimizer_tensor_owner_dtypes(tensor_entries, owner=owner, field=field)
    return {
        "owner_parameter_name": owner_name,
        "tensor_specs": [
            {
                "path": path,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": _tensor_sha256(tensor),
            }
            for (path, tensor) in tensor_entries
        ],
        "signature_sha256": _checkpoint_tree_sha256(value),
    }


def _validate_optimizer_state_record(
    value: object, *, record: object, owner: nn.Parameter, owner_name: str, field: str
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    _validate_checkpoint_tree(value, field)
    binding = _require_payload_mapping(
        record, expected_fields=_OPTIMIZER_STATE_RECORD_FIELDS, label=f"{field} binding"
    )
    if binding["owner_parameter_name"] != owner_name:
        raise ValueError(
            f"{field} owner parameter name does not match optimizer bindings"
        )
    expected_specs = binding["tensor_specs"]
    if not isinstance(expected_specs, list):
        raise ValueError(f"{field} binding tensor_specs must be a list")
    actual_entries = _optimizer_tensor_entries(value)
    if len(expected_specs) != len(actual_entries):
        raise ValueError(f"{field} tensor signature count does not match its binding")
    _validate_optimizer_tensor_owner_dtypes(actual_entries, owner=owner, field=field)
    for index, (expected, (actual_path, actual_tensor)) in enumerate(
        zip(expected_specs, actual_entries, strict=True)
    ):
        spec = _require_payload_mapping(
            expected,
            expected_fields=_OPTIMIZER_TENSOR_SPEC_FIELDS,
            label=f"{field} binding tensor_specs[{index}]",
        )
        if spec["path"] != actual_path:
            raise ValueError(f"{field} tensor path does not match its binding")
        if spec["shape"] != list(actual_tensor.shape):
            raise ValueError(f"{field} tensor shape does not match its binding")
        if spec["dtype"] != str(actual_tensor.dtype):
            raise ValueError(f"{field} tensor dtype does not match its binding")
        if spec["sha256"] != _tensor_sha256(actual_tensor):
            raise ValueError(f"{field} tensor signature does not match its binding")
    signature = binding["signature_sha256"]
    if not isinstance(signature, str) or len(signature) != 64:
        raise ValueError(
            f"{field} binding signature_sha256 must be a SHA-256 hex string"
        )
    if signature != _checkpoint_tree_sha256(value):
        raise ValueError(f"{field} signature does not match its binding")


def _optimizer_tensor_entries(
    value: object, path: str = "$"
) -> list[tuple[str, Tensor]]:
    if isinstance(value, Tensor):
        return [(path, value)]
    if isinstance(value, Mapping):
        entries: list[tuple[str, Tensor]] = []
        for key in sorted(value, key=_checkpoint_tree_key):
            entries.extend(_optimizer_tensor_entries(value[key], f"{path}[{key!r}]"))
        return entries
    if isinstance(value, list):
        return [
            entry
            for (index, item) in enumerate(value)
            for entry in _optimizer_tensor_entries(item, f"{path}[{index}]")
        ]
    if isinstance(value, tuple):
        return [
            entry
            for (index, item) in enumerate(value)
            for entry in _optimizer_tensor_entries(item, f"{path}({index})")
        ]
    return []


def _validate_optimizer_tensor_owner_dtypes(
    tensor_entries: list[tuple[str, Tensor]], *, owner: nn.Parameter, field: str
) -> None:
    for path, tensor in tensor_entries:
        if (
            tensor.is_floating_point()
            and tensor.shape == owner.shape
            and (tensor.dtype != owner.dtype)
        ):
            raise ValueError(
                f"{field}{path} floating tensor dtype {tensor.dtype} does not match owner parameter dtype {owner.dtype}"
            )


def _canonical_optimizer_group_hyperparameters(
    group: Mapping[object, object], field: str
) -> dict[str, object]:
    """Copy only canonical non-``params`` optimizer group fields for binding."""
    hyperparameters: dict[str, object] = {}
    for key, value in group.items():
        if not isinstance(key, str):
            raise ValueError(f"{field} keys must be strings")
        if key == "params":
            continue
        hyperparameters[key] = _canonical_optimizer_hyperparameter(
            value, f"{field}.{key}"
        )
    return dict(sorted(hyperparameters.items()))


def _canonical_optimizer_hyperparameter(value: object, field: str) -> object:
    """Validate standard scalar/sequence optimizer hyperparameters recursively."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if type(value) is int:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} must not be non-finite")
        return value
    if isinstance(value, tuple):
        return tuple(
            (
                _canonical_optimizer_hyperparameter(item, f"{field}[{index}]")
                for (index, item) in enumerate(value)
            )
        )
    if isinstance(value, list):
        return [
            _canonical_optimizer_hyperparameter(item, f"{field}[{index}]")
            for (index, item) in enumerate(value)
        ]
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field} mapping keys must be strings")
            normalized[key] = _canonical_optimizer_hyperparameter(
                item, f"{field}.{key}"
            )
        return dict(sorted(normalized.items()))
    raise ValueError(
        f"{field} must contain only canonical JSON scalar, list, tuple, or mapping values"
    )


def _validate_optimizer_group_hyperparameters(
    *,
    serialized_group: Mapping[object, object],
    group_binding: Mapping[str, object],
    group_index: int,
) -> None:
    field = f"optimizer_state.param_groups[{group_index}]"
    actual = _canonical_optimizer_group_hyperparameters(serialized_group, field)
    raw_expected = group_binding["hyperparameters"]
    if not isinstance(raw_expected, Mapping) or "params" in raw_expected:
        raise ValueError(
            f"optimizer_bindings.param_groups[{group_index}].hyperparameters is invalid"
        )
    expected = _canonical_optimizer_group_hyperparameters(
        raw_expected, f"optimizer_bindings.param_groups[{group_index}].hyperparameters"
    )
    if actual != expected:
        raise ValueError(
            f"{field} hyperparameters do not match their checkpoint binding"
        )
    signature = group_binding["hyperparameters_sha256"]
    if not isinstance(signature, str) or len(signature) != 64:
        raise ValueError(
            f"optimizer_bindings.param_groups[{group_index}].hyperparameters_sha256 must be a SHA-256 hex string"
        )
    if signature != _checkpoint_tree_sha256(expected):
        raise ValueError(f"{field} hyperparameter binding signature is invalid")
    if signature != _checkpoint_tree_sha256(actual):
        raise ValueError(f"{field} hyperparameter signature does not match its binding")
