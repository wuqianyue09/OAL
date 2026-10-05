"""Tensor, RNG and scheduler state serialization primitives."""

from __future__ import annotations
from collections.abc import Mapping
import hashlib
import math
from pathlib import Path
import random
import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer

_RNG_FIELDS = frozenset(("python", "numpy", "torch_cpu", "torch_cuda"))
_NUMPY_RNG_FIELDS = frozenset(
    ("bit_generator", "keys", "position", "has_gauss", "cached_gaussian")
)


def _tensor_sha256(tensor: Tensor) -> str:
    cpu_tensor = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(
        cpu_tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _checkpoint_tree_sha256(value: object) -> str:
    digest = hashlib.sha256()

    def update(item: object) -> None:
        if isinstance(item, Tensor):
            _hash_chunk(digest, b"tensor")
            _hash_chunk(digest, str(item.dtype).encode("utf-8"))
            _hash_chunk(digest, repr(tuple(item.shape)).encode("ascii"))
            _hash_chunk(
                digest,
                item.detach()
                .to(device="cpu")
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .numpy()
                .tobytes(),
            )
        elif item is None:
            _hash_chunk(digest, b"none")
        elif isinstance(item, bool):
            _hash_chunk(digest, b"bool:1" if item else b"bool:0")
        elif type(item) is int:
            _hash_chunk(digest, f"int:{item}".encode("ascii"))
        elif isinstance(item, float):
            _hash_chunk(digest, f"float:{item.hex()}".encode("ascii"))
        elif isinstance(item, str):
            _hash_chunk(digest, b"str")
            _hash_chunk(digest, item.encode("utf-8"))
        elif isinstance(item, Mapping):
            _hash_chunk(digest, b"mapping")
            for key in sorted(item, key=_checkpoint_tree_key):
                update(key)
                update(item[key])
        elif isinstance(item, list):
            _hash_chunk(digest, b"list")
            for child in item:
                update(child)
        elif isinstance(item, tuple):
            _hash_chunk(digest, b"tuple")
            for child in item:
                update(child)
        else:
            raise ValueError(
                f"optimizer state contains unsupported value {type(item).__name__}"
            )

    update(value)
    return digest.hexdigest()


def _hash_chunk(digest: object, value: bytes) -> None:
    if not isinstance(value, bytes):
        raise TypeError("checkpoint digest chunks must be bytes")
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def _checkpoint_tree_key(key: object) -> tuple[int, str]:
    if isinstance(key, str):
        return (0, key)
    if type(key) is int:
        return (1, str(key))
    raise ValueError(f"optimizer state has unsupported mapping key {key!r}")


def _snapshot_tensor_mapping(
    tensors: Mapping[str, Tensor], label: str
) -> dict[str, Tensor]:
    snapshot: dict[str, Tensor] = {}
    for name, tensor in tensors.items():
        _validate_live_tensor(tensor, f"{label}.{name}")
        snapshot[name] = tensor.detach().to(device="cpu").contiguous().clone()
    return snapshot


def _validate_tensor_mapping(
    payload_state: object,
    targets: Mapping[str, Tensor],
    label: str,
    *,
    immutable_names: frozenset[str] = frozenset(),
) -> list[tuple[Tensor, Tensor]]:
    if not isinstance(payload_state, Mapping):
        raise ValueError(f"{label} must be a mapping")
    actual_names = set(payload_state)
    expected_names = set(targets)
    unknown = sorted(
        (name for name in actual_names - expected_names if isinstance(name, str))
    )
    non_string = sorted(
        (repr(name) for name in actual_names if not isinstance(name, str))
    )
    missing = sorted(expected_names - actual_names)
    if unknown or non_string:
        detail = unknown + non_string
        raise ValueError(f"{label} has unknown key(s): {', '.join(detail)}")
    if missing:
        raise ValueError(f"{label} is missing key(s): {', '.join(missing)}")
    if not immutable_names.issubset(expected_names):
        raise ValueError(
            f"{label} immutable tensor allow-list is not a subset of targets"
        )
    actions: list[tuple[Tensor, Tensor]] = []
    for name, target in targets.items():
        source = payload_state[name]
        if not isinstance(source, Tensor):
            raise ValueError(f"{label}.{name} must be a torch.Tensor")
        if source.device.type != "cpu":
            raise ValueError(f"{label}.{name} must be stored on CPU")
        if source.shape != target.shape:
            raise ValueError(
                f"{label}.{name} shape {tuple(source.shape)} does not match target shape {tuple(target.shape)}"
            )
        if source.dtype != target.dtype:
            raise ValueError(
                f"{label}.{name} dtype {source.dtype} does not match target dtype {target.dtype}"
            )
        _validate_live_tensor(source, f"{label}.{name}")
        if name in immutable_names:
            target_cpu = target.detach().to(device="cpu").contiguous()
            if not torch.equal(source, target_cpu):
                raise ValueError(
                    f"{label}.{name} is immutable kernel metadata and does not match the live parameter-bank contract"
                )
            continue
        actions.append((target, source))
    return actions


def _copy_actions(actions: list[tuple[Tensor, Tensor]]) -> None:
    """Copy only after all state fields have passed schema/shape/dtype checks."""
    with torch.no_grad():
        for target, source in actions:
            target.copy_(source.to(device=target.device), non_blocking=False)


def _checkpoint_path(
    run_dir: str | Path, filename: str, *, require_file: bool = False
) -> Path:
    path = Path(run_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint run directory does not exist: {path}")
    destination = path / filename
    if require_file and (not destination.is_file()):
        raise FileNotFoundError(f"checkpoint file does not exist: {destination}")
    return destination


def _safe_load(path: Path) -> dict[str, object]:
    """Use PyTorch's weights-only loader and turn unsafe payloads into clear errors."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"checkpoint could not be safely loaded from {path}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint payload must be a plain dictionary: {path}")
    return payload


def _require_payload_mapping(
    payload: object, *, expected_fields: frozenset[str], label: str
) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} payload must be a plain dictionary")
    actual_fields = set(payload)
    unknown = sorted(
        (field for field in actual_fields - expected_fields if isinstance(field, str))
    )
    non_string = sorted(
        (repr(field) for field in actual_fields if not isinstance(field, str))
    )
    missing = sorted(expected_fields - actual_fields)
    if unknown or non_string:
        raise ValueError(
            f"{label} has unknown field(s): {', '.join(unknown + non_string)}"
        )
    if missing:
        raise ValueError(f"{label} is missing required field(s): {', '.join(missing)}")
    return payload


def _payload_step(payload: Mapping[str, object]) -> int:
    return _require_step(payload["step"])


def _payload_nll(payload: Mapping[str, object]) -> float:
    return _require_finite_number(payload["validation_nll"], "validation_nll")


def _require_step(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("step must be a non-negative integer")
    return value


def _require_finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite Python number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _validate_live_tensor(tensor: Tensor, field: str) -> None:
    if tensor.is_floating_point() or tensor.is_complex():
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{field} contains non-finite tensor values")


def _clone_checkpoint_tree(value: object, field: str) -> object:
    """Clone a weights-only-safe optimizer/scheduler tree onto CPU."""
    if isinstance(value, Tensor):
        _validate_live_tensor(value, field)
        return value.detach().to(device="cpu").contiguous().clone()
    if value is None or isinstance(value, (bool, str)):
        return value
    if type(value) is int:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} must not contain non-finite scalar values")
        return value
    if isinstance(value, Mapping):
        clone: dict[str | int, object] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise ValueError(f"{field} has unsupported mapping key {key!r}")
            clone[key] = _clone_checkpoint_tree(item, f"{field}[{key!r}]")
        return clone
    if isinstance(value, list):
        return [
            _clone_checkpoint_tree(item, f"{field}[{index}]")
            for (index, item) in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            (
                _clone_checkpoint_tree(item, f"{field}[{index}]")
                for (index, item) in enumerate(value)
            )
        )
    raise ValueError(
        f"{field} contains unsupported checkpoint value {type(value).__name__}"
    )


def _validate_checkpoint_tree(value: object, field: str) -> None:
    _clone_checkpoint_tree(value, field)


def _scheduler_state_for_save(scheduler: object) -> object:
    state_dict = getattr(scheduler, "state_dict", None)
    if not callable(state_dict):
        raise TypeError("scheduler must provide a callable state_dict() method")
    return _clone_checkpoint_tree(state_dict(), "scheduler_state")


def _validate_scheduler_state(payload_state: object, scheduler: object | None) -> None:
    if payload_state is None:
        if scheduler is not None:
            raise ValueError(
                "resume checkpoint has no scheduler_state but a scheduler was supplied"
            )
        return
    if scheduler is None:
        raise ValueError("resume checkpoint requires a scheduler but none was supplied")
    if not callable(getattr(scheduler, "load_state_dict", None)):
        raise TypeError("scheduler must provide a callable load_state_dict() method")
    _validate_checkpoint_tree(payload_state, "scheduler_state")


def _load_scheduler_state(payload_state: object, scheduler: object | None) -> None:
    if payload_state is None:
        assert scheduler is None
        return
    assert scheduler is not None
    try:
        scheduler.load_state_dict(payload_state)
    except Exception as exc:
        raise ValueError(
            f"scheduler_state is incompatible with the supplied scheduler: {exc}"
        ) from exc


def _capture_rng_state() -> dict[str, object]:
    numpy_state = np.random.get_state()
    if len(numpy_state) != 5 or not isinstance(numpy_state[0], str):
        raise RuntimeError("NumPy returned an unsupported legacy RNG state")
    numpy_keys = numpy_state[1]
    if not isinstance(numpy_keys, np.ndarray) or numpy_keys.dtype != np.uint32:
        raise RuntimeError("NumPy returned an unsupported RNG key array")
    cuda_state: list[Tensor] | None = None
    if torch.cuda.is_available():
        cuda_state = [
            state.detach().to(device="cpu").clone()
            for state in torch.cuda.get_rng_state_all()
        ]
    return {
        "python": _clone_checkpoint_tree(random.getstate(), "rng_state.python"),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_keys.copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().detach().to(device="cpu").clone(),
        "torch_cuda": cuda_state,
    }


def _validate_rng_state(value: object) -> None:
    record = _require_payload_mapping(
        value, expected_fields=_RNG_FIELDS, label="rng_state"
    )
    _validate_checkpoint_tree(record["python"], "rng_state.python")
    numpy_state = _require_payload_mapping(
        record["numpy"], expected_fields=_NUMPY_RNG_FIELDS, label="rng_state.numpy"
    )
    if not isinstance(numpy_state["bit_generator"], str):
        raise ValueError("rng_state.numpy.bit_generator must be a string")
    keys = numpy_state["keys"]
    if (
        not isinstance(keys, Tensor)
        or keys.device.type != "cpu"
        or keys.dtype != torch.uint32
        or (keys.ndim != 1)
    ):
        raise ValueError("rng_state.numpy.keys must be a CPU uint32 vector")
    for field in ("position", "has_gauss"):
        if type(numpy_state[field]) is not int:
            raise ValueError(f"rng_state.numpy.{field} must be an integer")
    _require_finite_number(
        numpy_state["cached_gaussian"], "rng_state.numpy.cached_gaussian"
    )
    cpu_state = record["torch_cpu"]
    if (
        not isinstance(cpu_state, Tensor)
        or cpu_state.device.type != "cpu"
        or cpu_state.dtype != torch.uint8
        or (cpu_state.ndim != 1)
    ):
        raise ValueError("rng_state.torch_cpu must be a CPU uint8 vector")
    cuda_state = record["torch_cuda"]
    cuda_available = bool(torch.cuda.is_available())
    if cuda_state is None:
        if cuda_available:
            raise ValueError("rng_state.torch_cuda is required when CUDA is available")
        return
    if not isinstance(cuda_state, list):
        raise ValueError("rng_state.torch_cuda must be null or a list of CPU tensors")
    if not cuda_available:
        raise ValueError("checkpoint includes CUDA RNG state but CUDA is not available")
    try:
        device_count = int(torch.cuda.device_count())
    except Exception as exc:
        raise ValueError(
            f"rng_state.torch_cuda device_count could not be queried: {exc}"
        ) from exc
    if len(cuda_state) != device_count:
        raise ValueError(
            f"rng_state.torch_cuda list length must equal current CUDA device_count ({len(cuda_state)} != {device_count})"
        )
    for index, state in enumerate(cuda_state):
        if (
            not isinstance(state, Tensor)
            or state.device.type != "cpu"
            or state.dtype != torch.uint8
        ):
            raise ValueError(
                f"rng_state.torch_cuda[{index}] must be a CPU uint8 tensor"
            )


def _restore_rng_state(value: object) -> None:
    _validate_rng_state(value)
    assert isinstance(value, Mapping)
    python_state = value["python"]
    try:
        random.setstate(python_state)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"rng_state.python is invalid: {exc}") from exc
    numpy_state = value["numpy"]
    assert isinstance(numpy_state, Mapping)
    keys = numpy_state["keys"]
    assert isinstance(keys, Tensor)
    try:
        np.random.set_state(
            (
                numpy_state["bit_generator"],
                keys.numpy().astype(np.uint32, copy=True),
                numpy_state["position"],
                numpy_state["has_gauss"],
                numpy_state["cached_gaussian"],
            )
        )
    except Exception as exc:
        raise ValueError(f"rng_state.numpy is invalid: {exc}") from exc
    cpu_state = value["torch_cpu"]
    assert isinstance(cpu_state, Tensor)
    torch.set_rng_state(cpu_state)
    cuda_state = value["torch_cuda"]
    if cuda_state is None:
        return
    assert isinstance(cuda_state, list)
    try:
        torch.cuda.set_rng_state_all(cuda_state)
    except Exception as exc:
        raise ValueError(
            f"rng_state.torch_cuda is invalid for current CUDA devices: {exc}"
        ) from exc


def _move_optimizer_state_to_owner_devices(optimizer: Optimizer) -> None:
    for parameter, state in optimizer.state.items():
        if not isinstance(parameter, Tensor):
            raise ValueError("optimizer state has a non-tensor parameter key")
        optimizer.state[parameter] = _move_tree_to_device(state, parameter.device)


def _move_tree_to_device(value: object, device: torch.device) -> object:
    if isinstance(value, Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {
            key: _move_tree_to_device(item, device) for (key, item) in value.items()
        }
    if isinstance(value, list):
        return [_move_tree_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple((_move_tree_to_device(item, device) for item in value))
    return value
