"""Trainable OAL factors and persistent grouping state for Qwen LoRA.

This module deliberately creates all initial state on the CPU.  The model
setup layer owns the later module-wide move to the selected CUDA device, which
keeps parameters and persistent buffers together without making local tests
depend on CUDA.
"""

from __future__ import annotations
import hashlib
import math
from typing import cast
import torch
from torch import Tensor, nn
from .grouped_initialization import (
    GroupedParameterInitialization,
    build_grouped_dim_groups,
)
from .backbones.spec import ModelGeometry
from .config import GROUPED_BASE_METHODS, METHOD_NAMES, MethodName
from .experiment_contract import (
    NUM_LAYERS,
    NUM_QUERY_HEADS,
    HEAD_DIM,
    REPLACEMENT_LAYER_IDS,
)

GROUPED_MAX_GROUP_COUNT = 8
GROUPED_FACTOR_SIZE = (GROUPED_MAX_GROUP_COUNT + 1) * (GROUPED_MAX_GROUP_COUNT + 2) // 2
LEGACY_BINARY_GROUPED_FACTOR_SIZE = 6
DEFAULT_KERNEL_EPSILON = 1e-06
DEFAULT_TRAINABLE_KERNEL_LAYER_IDS = REPLACEMENT_LAYER_IDS
ADAPTIVE_RUNTIME_GROUP_COUNTS = frozenset(range(2, 6))
QWEN_GEOMETRY = ModelGeometry(NUM_LAYERS, 896, NUM_QUERY_HEADS, 2, HEAD_DIM)


def tensor_checksum(tensor: Tensor) -> str:
    """Return the SHA-256 checksum of exact contiguous CPU tensor bytes."""
    if not isinstance(tensor, Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    raw = (
        tensor.detach()
        .to(device="cpu")
        .contiguous()
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )
    return hashlib.sha256(raw).hexdigest()


class KernelParameterBank(nn.Module):
    """Register one FP32 factor per selected layer and head.

    Gmax=8 packing is built differentiably at runtime. Inactive padding and
    unselected layers have no trainable leaves or optimizer state.
    """

    def __init__(
        self,
        method: MethodName | str,
        *,
        grouped_initialization: GroupedParameterInitialization,
        grouped_kernel_epsilon: float = DEFAULT_KERNEL_EPSILON,
        trainable_kernel_layer_ids: tuple[
            int, ...
        ] = DEFAULT_TRAINABLE_KERNEL_LAYER_IDS,
        geometry: ModelGeometry = QWEN_GEOMETRY,
    ) -> None:
        super().__init__()
        self.geometry = _require_geometry(geometry)
        self.method = _require_method(method)
        grouped_epsilon = _require_positive_finite_scalar(
            grouped_kernel_epsilon, "grouped_kernel_epsilon"
        )
        selected_kernel_layers = _require_trainable_kernel_layer_ids(
            trainable_kernel_layer_ids, self.geometry
        )
        self._active_kernel_layer_ids = selected_kernel_layers
        if not isinstance(grouped_initialization, GroupedParameterInitialization):
            raise TypeError(
                "grouped_initialization must contain runtime OAL parameters"
            )
        asset = _require_grouped_parameter_state(
            grouped_initialization, geometry=self.geometry
        )
        if asset.layer_ids_zero_based != selected_kernel_layers:
            raise ValueError(
                "OAL parameter layers must exactly match trainable_kernel_layer_ids"
            )
        self._initialize_adaptive_grouped_state(asset)
        stored_epsilon = _require_exact_grouped_epsilon_representation(
            grouped_epsilon, asset.epsilon
        )
        self.register_buffer(
            "grouped_kernel_epsilon",
            torch.tensor(stored_epsilon, dtype=torch.float32),
            persistent=True,
        )
        self._grouped_kernel_epsilon_runtime = stored_epsilon
        self.validate_grouped_static_contract()
        self._initialization_checksum = _adaptive_grouped_initialization_checksum(self)
        self.register_load_state_dict_post_hook(self._validate_grouped_state_after_load)

    @property
    def initialization_checksum(self) -> str | None:
        """The recorded checksum of the bank's initialized or restored state."""
        return self._initialization_checksum

    @property
    def active_kernel_layer_ids(self) -> tuple[int, ...]:
        """Return the layers with trainable OAL parameters."""
        return self._active_kernel_layer_ids

    def _initialize_adaptive_grouped_state(
        self, asset: GroupedParameterInitialization
    ) -> None:
        """Register selected-head factors without trainable Gmax padding."""
        layer_ids = asset.layer_ids_zero_based
        factors = nn.ModuleDict()
        for layer_id in layer_ids:
            per_head = nn.ParameterDict()
            for head_id in range(self.geometry.num_query_heads):
                record = asset.heads[layer_id, head_id]
                active = torch.tensor(
                    record.packed_lower_triangular, dtype=torch.float32, device="cpu"
                ).contiguous()
                per_head[f"head_{head_id}"] = nn.Parameter(active)
            factors[f"layer_{layer_id}"] = per_head
        self.grouped_active_factors = factors
        self.register_buffer(
            "grouped_layer_ids",
            torch.tensor(layer_ids, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "dim_groups",
            torch.from_numpy(build_grouped_dim_groups(asset))
            .to(dtype=torch.int32)
            .contiguous(),
            persistent=True,
        )
        self.register_buffer(
            "group_counts",
            torch.tensor(
                [
                    [
                        asset.heads[layer_id, head_id].group_count
                        for head_id in range(self.geometry.num_query_heads)
                    ]
                    for layer_id in layer_ids
                ],
                dtype=torch.int32,
            ).contiguous(),
            persistent=True,
        )
        self._grouped_trainable_layer_ids = tuple(layer_ids)
        self._grouped_layer_offsets = {
            layer_id: offset for (offset, layer_id) in enumerate(layer_ids)
        }

    @property
    def trainable_kernel_layer_ids(self) -> tuple[int, ...]:
        """Return exactly the model layers owning grouped active parameters."""
        if self.method in GROUPED_BASE_METHODS:
            return self._grouped_trainable_layer_ids
        return ()

    def parameters_for_layer(self, layer_id: int) -> Tensor:
        """Return packed OAL factors for the selected layer.

        Adaptive factors are assembled from per-head active Parameters and
        constant zero padding at every call.  The returned ``[14, 45]`` tensor
        is therefore differentiable, while inactive Gmax slots do not appear
        in ``named_parameters()`` or any optimizer input list.
        """
        _require_layer_id(layer_id)
        if self.method in GROUPED_BASE_METHODS:
            return self.packed_grouped_factor_for_layer(layer_id)
        raise RuntimeError(f"{self.method} does not have trainable kernel parameters")

    layer_parameters = parameters_for_layer

    def active_grouped_factor_for_layer_head(
        self, layer_id: int, head_id: int
    ) -> nn.Parameter:
        """Return one independent active-only FP32 grouped factor Parameter."""
        _require_layer_id(layer_id, self.geometry)
        _require_head_id(head_id, self.geometry)
        if self.method not in GROUPED_BASE_METHODS:
            raise RuntimeError(f"{self.method} does not have grouped factors")
        self._require_selected_grouped_layer(layer_id)
        parameter = self.grouped_active_factors[f"layer_{layer_id}"][f"head_{head_id}"]
        if not isinstance(parameter, nn.Parameter):
            raise RuntimeError("grouped active factor registration is corrupted")
        return parameter

    def grouped_active_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return active grouped leaves in canonical layer/head order only."""
        if self.method not in GROUPED_BASE_METHODS:
            return ()
        return tuple(
            (
                self.active_grouped_factor_for_layer_head(layer_id, head_id)
                for layer_id in self.trainable_kernel_layer_ids
                for head_id in range(self.geometry.num_query_heads)
            )
        )

    def packed_grouped_factor_for_layer(self, layer_id: int) -> Tensor:
        """Differentiably pack one selected layer into public ``[14, 45]`` FP32."""
        _require_layer_id(layer_id)
        if self.method not in GROUPED_BASE_METHODS:
            raise RuntimeError(f"{self.method} does not have grouped factors")
        self._require_selected_grouped_layer(layer_id)
        per_head = self.grouped_active_factors[f"layer_{layer_id}"]
        packed_heads: list[Tensor] = []
        for head_id in range(self.geometry.num_query_heads):
            active = per_head[f"head_{head_id}"]
            padding_length = GROUPED_FACTOR_SIZE - active.numel()
            padding = torch.zeros(
                padding_length,
                dtype=torch.float32,
                device=active.device,
                requires_grad=False,
            )
            packed_heads.append(torch.cat((active, padding), dim=0))
        packed = torch.stack(packed_heads, dim=0).contiguous()
        if (
            packed.shape != (self.geometry.num_query_heads, GROUPED_FACTOR_SIZE)
            or packed.dtype is not torch.float32
        ):
            raise RuntimeError(
                "adaptive grouped factor packing produced an invalid public factor"
            )
        return packed

    def dim_groups_for_layer(self, layer_id: int) -> Tensor:
        """Return the frozen `[14, 64]` labels for one selected grouped layer."""
        _require_layer_id(layer_id)
        if self.method not in GROUPED_BASE_METHODS:
            raise RuntimeError(f"{self.method} does not have grouped dimension labels")
        return self.dim_groups[self._grouped_layer_offset(layer_id)]

    def group_counts_for_layer(self, layer_id: int) -> Tensor:
        """Return per-head true group counts (2..5), not padded Gmax labels."""
        _require_layer_id(layer_id)
        if self.method not in GROUPED_BASE_METHODS:
            raise RuntimeError(f"{self.method} does not have grouped dimension labels")
        return self.group_counts[self._grouped_layer_offset(layer_id)]

    def grouped_kernel_epsilon_for_runtime(self) -> float:
        """Return the construction-validated exact FP32 epsilon without sync."""
        if self.method not in GROUPED_BASE_METHODS:
            raise RuntimeError(f"{self.method} does not have a grouped kernel epsilon")
        return self._grouped_kernel_epsilon_runtime

    def validate_grouped_static_contract(self) -> None:
        """Synchronously validate immutable grouped metadata outside forward.

        Bank construction, adapter construction, and grouped checkpoint load
        are admission points.  Detailed per-head scans and Python scalar
        conversion belong there; the attention hot path only packs the active
        factor, retrieves its labels, and checks cheap tensor properties.
        """
        if self.method not in GROUPED_BASE_METHODS:
            raise RuntimeError(f"{self.method} does not have grouped runtime state")
        layer_ids = self._grouped_trainable_layer_ids
        if not layer_ids:
            raise ValueError(
                "grouped parameter bank must contain selected adaptive layers"
            )
        serialized_grouped_layer_ids = self._frozen_runtime_buffer("grouped_layer_ids")
        groups = self._frozen_runtime_buffer("dim_groups")
        counts = self._frozen_runtime_buffer("group_counts")
        epsilon = self._frozen_runtime_buffer("grouped_kernel_epsilon")
        if (
            groups.shape
            != (len(layer_ids), self.geometry.num_query_heads, self.geometry.head_dim)
            or groups.dtype is not torch.int32
            or (not groups.is_contiguous())
            or (groups.device.type == "meta")
        ):
            raise ValueError(
                "dim_groups must be contiguous int32 selected-layer metadata"
            )
        if (
            counts.shape != (len(layer_ids), self.geometry.num_query_heads)
            or counts.dtype is not torch.int32
            or counts.device != groups.device
        ):
            raise ValueError(
                "group_counts must be int32 selected-layer metadata on the groups device"
            )
        if (
            epsilon.shape != ()
            or epsilon.dtype is not torch.float32
            or epsilon.device != groups.device
            or (not bool(torch.isfinite(epsilon).all().item()))
            or (not bool((epsilon > 0).item()))
        ):
            raise ValueError(
                "grouped_kernel_epsilon must be a finite positive FP32 scalar"
            )
        stored_epsilon = float(epsilon.detach().to(device="cpu", dtype=torch.float32))
        if stored_epsilon != self._grouped_kernel_epsilon_runtime:
            raise ValueError(
                "grouped_kernel_epsilon disagrees with its construction-validated FP32 value"
            )
        serialized_layer_ids = tuple(
            (
                int(value)
                for value in serialized_grouped_layer_ids.detach().cpu().tolist()
            )
        )
        if serialized_layer_ids != layer_ids:
            raise ValueError(
                "grouped layer IDs disagree with the active parameter mapping"
            )
        for layer_offset, layer_id in enumerate(layer_ids):
            for head_id in range(self.geometry.num_query_heads):
                group_count = int(counts[layer_offset, head_id].item())
                if group_count not in ADAPTIVE_RUNTIME_GROUP_COUNTS:
                    raise ValueError(
                        f"group_counts layer={layer_id} head={head_id} must be in {tuple(sorted(ADAPTIVE_RUNTIME_GROUP_COUNTS))}"
                    )
                labels = groups[layer_offset, head_id]
                expected_labels = torch.arange(
                    group_count, dtype=labels.dtype, device=labels.device
                )
                if not torch.equal(torch.unique(labels, sorted=True), expected_labels):
                    raise ValueError(
                        f"dim_groups layer={layer_id} head={head_id} must form a continuous prefix 0..{group_count - 1} with every group present"
                    )
                active = self.active_grouped_factor_for_layer_head(layer_id, head_id)
                expected_factor_length = (group_count + 1) * (group_count + 2) // 2
                if (
                    active.numel() != expected_factor_length
                    or active.dtype is not torch.float32
                    or (not active.requires_grad)
                    or (not active.is_leaf)
                    or (not active.is_contiguous())
                    or (active.device != groups.device)
                    or (not bool(torch.isfinite(active).all().item()))
                ):
                    raise ValueError(
                        f"active grouped factor layer={layer_id} head={head_id} does not match its true group prefix"
                    )

    def _frozen_runtime_buffer(self, name: str) -> Tensor:
        """Return one non-trainable local buffer or reject registration drift."""
        buffer = self._buffers.get(name)
        if (
            not isinstance(buffer, Tensor)
            or name in self._parameters
            or buffer.requires_grad
        ):
            raise ValueError(f"{name} must remain a registered frozen runtime buffer")
        return buffer

    def _grouped_layer_offset(self, layer_id: int) -> int:
        self._require_selected_grouped_layer(layer_id)
        return self._grouped_layer_offsets[layer_id]

    def _require_selected_grouped_layer(self, layer_id: int) -> None:
        if layer_id not in self._grouped_layer_offsets:
            raise ValueError(
                f"layer_id {layer_id} is not a selected adaptive layer; selected={self.trainable_kernel_layer_ids}"
            )

    def _validate_grouped_state_after_load(
        self, module: nn.Module, _incompatible_keys: object
    ) -> None:
        """Validate grouped leaves after recursive state loading has completed."""
        if module is not self:
            raise RuntimeError("grouped state post-load hook received the wrong module")
        self.validate_grouped_static_contract()

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Reject checkpoint attempts to rewrite immutable kernel metadata."""
        immutable_mismatch = False
        immutable_by_method: dict[MethodName, tuple[str, ...]] = {
            "grouped_quadratic": (
                "grouped_layer_ids",
                "dim_groups",
                "group_counts",
                "grouped_kernel_epsilon",
            )
        }
        for name in immutable_by_method[self.method]:
            incoming = state_dict.get(prefix + name)
            current = getattr(self, name)
            if (
                not isinstance(incoming, Tensor)
                or isinstance(incoming, nn.Parameter)
                or incoming.requires_grad
                or (incoming.shape != current.shape)
                or (incoming.dtype != current.dtype)
                or (
                    not torch.equal(
                        incoming.detach().to(device="cpu"),
                        current.detach().to(device="cpu"),
                    )
                )
            ):
                immutable_mismatch = True
                error_msgs.append(
                    f"{prefix}{name} does not match immutable kernel runtime metadata"
                )
        if immutable_mismatch:
            raise RuntimeError("\n".join(error_msgs))
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _apply(self, fn: object, recurse: bool = True) -> KernelParameterBank:
        """Move state without downcasting FP32 master parameters or buffers.

        ``nn.Module.to(dtype=...)`` normally applies its conversion callable to
        every floating parameter and buffer. OAL state stays FP32 when the enclosing
        Qwen model is BF16, so this override applies only the callable's device
        migration to floating state while preserving its FP32 dtype.
        """
        if not callable(fn):
            raise TypeError(
                "KernelParameterBank._apply requires a tensor conversion callable"
            )
        _apply_module_state_preserving_fp32(self, fn, recurse=recurse)
        return self


def _require_grouped_parameter_state(
    asset: GroupedParameterInitialization, *, geometry: ModelGeometry
) -> GroupedParameterInitialization:
    """Check the numerical state loaded from the released OAL asset."""
    if asset.model_num_layers != geometry.num_layers:
        raise ValueError(
            f"adaptive_group_asset model_num_layers must be {geometry.num_layers}"
        )
    if (
        asset.num_query_heads != geometry.num_query_heads
        or asset.head_dim != geometry.head_dim
    ):
        raise ValueError(
            "adaptive_group_asset does not match the grouped runtime shape"
        )
    layer_ids = asset.layer_ids_zero_based
    if not layer_ids:
        raise ValueError(
            "adaptive_group_asset must declare at least one selected layer"
        )
    if tuple(sorted(layer_ids)) != layer_ids or len(set(layer_ids)) != len(layer_ids):
        raise ValueError(
            "adaptive_group_asset layer_ids_zero_based must be strictly increasing"
        )
    for layer_id in layer_ids:
        _require_layer_id(layer_id, geometry)
        for head_id in range(geometry.num_query_heads):
            record = asset.heads.get((layer_id, head_id))
            if record is None:
                raise ValueError(
                    f"adaptive_group_asset is missing selected record ({layer_id}, {head_id})"
                )
            if not 2 <= record.group_count <= 5:
                raise ValueError("adaptive_group_asset group_count must be in 2..5")
            if len(record.groups) != record.group_count:
                raise ValueError(
                    "adaptive_group_asset group partition must match group_count"
                )
            covered_dimensions: set[int] = set()
            for group_id, dimensions in enumerate(record.groups):
                if not dimensions:
                    raise ValueError(
                        "adaptive_group_asset group partition must have every true prefix group present"
                    )
                if any(
                    (
                        type(dimension) is not int
                        or dimension < 0
                        or dimension >= geometry.head_dim
                        for dimension in dimensions
                    )
                ):
                    raise ValueError(
                        "adaptive_group_asset group partition contains invalid dimensions"
                    )
                if tuple(sorted(dimensions)) != dimensions or len(
                    set(dimensions)
                ) != len(dimensions):
                    raise ValueError(
                        "adaptive_group_asset group partition must be sorted and unique"
                    )
                overlap = covered_dimensions & set(dimensions)
                if overlap:
                    raise ValueError(
                        f"adaptive_group_asset group partition has duplicate dimensions: {sorted(overlap)!r}"
                    )
                covered_dimensions.update(dimensions)
            if covered_dimensions != set(range(geometry.head_dim)):
                raise ValueError(
                    "adaptive_group_asset group partition must cover dimensions 0..63 exactly"
                )
            expected_length = (record.group_count + 1) * (record.group_count + 2) // 2
            if len(record.packed_lower_triangular) != expected_length:
                raise ValueError(
                    "adaptive_group_asset active factor has an invalid packed length"
                )
            if expected_length >= GROUPED_FACTOR_SIZE:
                raise ValueError(
                    "adaptive_group_asset active factor must fit Gmax=8 packing"
                )
            if not all(
                (math.isfinite(value) for value in record.packed_lower_triangular)
            ):
                raise ValueError("adaptive_group_asset active factors must be finite")
    expected_keys = {
        (layer_id, head_id)
        for layer_id in layer_ids
        for head_id in range(geometry.num_query_heads)
    }
    if set(asset.heads) != expected_keys:
        raise ValueError(
            "adaptive_group_asset must not contain unselected factor records"
        )
    _require_positive_finite_scalar(asset.epsilon, "adaptive_group_asset epsilon")
    return asset


def _adaptive_grouped_initialization_checksum(bank: KernelParameterBank) -> str:
    """Hash active factor bytes plus their global layer/head mapping."""
    digest = hashlib.sha256()
    for layer_id in bank.trainable_kernel_layer_ids:
        for head_id in range(bank.geometry.num_query_heads):
            digest.update(f"{layer_id}:{head_id}:".encode("ascii"))
            active = bank.active_grouped_factor_for_layer_head(layer_id, head_id)
            digest.update(
                active.detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
                .numpy()
                .tobytes()
            )
    return digest.hexdigest()


def _require_exact_grouped_epsilon_representation(
    requested: float, asset_epsilon: float
) -> float:
    """Return the common FP32 scalar or reject a one-ULP asset mismatch.

    Configuration literals such as ``1e-6`` cannot be compared as Python
    binary64 values to a checkpoint FP32 buffer.  Equality of their *stored
    FP32 representations* is both reproducible and exact: values that round
    to the same FP32 scalar are allowed, while a meaningful one-ULP change is
    rejected before the bank exists.
    """
    requested_fp32 = torch.tensor(requested, dtype=torch.float32)
    asset_fp32 = torch.tensor(asset_epsilon, dtype=torch.float32)
    if not torch.equal(requested_fp32, asset_fp32):
        raise ValueError(
            "grouped_kernel_epsilon must have the canonical adaptive asset's exact FP32 representation"
        )
    return float(asset_fp32)


def _require_method(method: MethodName | str) -> MethodName:
    if not isinstance(method, str) or method not in METHOD_NAMES:
        raise ValueError(f"method must be one of {METHOD_NAMES}")
    return cast(MethodName, method)


def _require_kernel_epsilon(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite Python scalar")
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 < resolved < 0.5:
        raise ValueError(f"{name} must be finite, positive, and less than 0.5")
    return resolved


def _require_positive_finite_scalar(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite Python scalar")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    fp32_value = torch.tensor(resolved, dtype=torch.float32)
    if not bool(torch.isfinite(fp32_value)) or not bool(fp32_value > 0.0):
        raise ValueError(f"{name} must remain finite and positive when stored as FP32")
    return resolved


def _move_tensor_preserving_fp32(tensor: Tensor, fn: object) -> Tensor:
    """Apply a module conversion's device target while retaining FP32 data."""
    if not callable(fn):
        raise TypeError("tensor conversion callable must be callable")
    with torch.no_grad():
        if tensor.device.type == "meta":
            return fn(tensor).to(dtype=torch.float32)
        device_probe = torch.empty(0, dtype=tensor.dtype, device=tensor.device)
        target_device = fn(device_probe).device
        return tensor.to(device=target_device, dtype=torch.float32)


def _apply_module_state_preserving_fp32(
    module: nn.Module, fn: object, *, recurse: bool
) -> None:
    """Recursively move a module tree without letting nested containers downcast.

    ``ModuleDict``/``ParameterDict`` inherit PyTorch's default ``_apply`` and
    otherwise receive the BF16 conversion before the parent bank can restore
    their leaves.  Walking the tree ourselves applies the same device move to
    every nested active factor while retaining FP32 master storage.
    """
    if not callable(fn):
        raise TypeError("module conversion callable must be callable")
    if recurse:
        for child in module.children():
            _apply_module_state_preserving_fp32(child, fn, recurse=True)
    for parameter_name, parameter in tuple(module._parameters.items()):
        if parameter is None:
            continue
        converted = _move_tensor_preserving_fp32(parameter, fn)
        original_grad = parameter.grad
        if converted.device == parameter.device:
            parameter.data = converted
            output_parameter = parameter
        else:
            output_parameter = nn.Parameter(
                converted, requires_grad=parameter.requires_grad
            )
            module._parameters[parameter_name] = output_parameter
        if original_grad is not None:
            output_parameter.grad = _move_tensor_preserving_fp32(original_grad, fn)
    for buffer_name, buffer in module._buffers.items():
        if buffer is None:
            continue
        module._buffers[buffer_name] = (
            _move_tensor_preserving_fp32(buffer, fn)
            if buffer.is_floating_point()
            else fn(buffer)
        )


def _require_geometry(geometry: object) -> ModelGeometry:
    if not isinstance(geometry, ModelGeometry):
        raise TypeError("geometry must be a ModelGeometry")
    if geometry.head_dim != HEAD_DIM:
        raise ValueError(f"kernel geometry head_dim must be exactly {HEAD_DIM}")
    if (
        geometry.num_layers <= 0
        or geometry.hidden_size <= 0
        or geometry.num_query_heads <= 0
        or (geometry.num_kv_heads <= 0)
        or geometry.num_query_heads % geometry.num_kv_heads
        or (geometry.hidden_size != geometry.num_query_heads * geometry.head_dim)
    ):
        raise ValueError("kernel geometry is internally inconsistent")
    return geometry


def _require_layer_id(layer_id: int, geometry: ModelGeometry = QWEN_GEOMETRY) -> None:
    if type(layer_id) is not int or not 0 <= layer_id < geometry.num_layers:
        raise ValueError(f"layer_id must be an integer in 0..{geometry.num_layers - 1}")


def _require_head_id(head_id: int, geometry: ModelGeometry = QWEN_GEOMETRY) -> None:
    if type(head_id) is not int or not 0 <= head_id < geometry.num_query_heads:
        raise ValueError(
            f"head_id must be an integer in 0..{geometry.num_query_heads - 1}"
        )


def _require_trainable_kernel_layer_ids(
    value: object, geometry: ModelGeometry = QWEN_GEOMETRY
) -> tuple[int, ...]:
    """Validate the exact active layer set without materializing inactive leaves."""
    if not isinstance(value, tuple) or any(
        (type(layer_id) is not int for layer_id in value)
    ):
        raise TypeError(
            "trainable_kernel_layer_ids must be a tuple of integer layer IDs"
        )
    if not value:
        raise ValueError("trainable_kernel_layer_ids must not be empty")
    if tuple(sorted(set(value))) != value:
        raise ValueError("trainable_kernel_layer_ids must be strictly increasing")
    for layer_id in value:
        _require_layer_id(layer_id, geometry)
    return value
