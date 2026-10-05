"""Strict common adapter for the measured Qwen2.5-0.5B attention contract.

This module is intentionally narrow.  It moves the four projection modules
from a measured ``Qwen2Attention`` into a replacement module, validates the
no-cache causal pilot inputs, applies the model-owned RoPE function, and lets
one method-specific subclass provide only ``head_attention``.  It never calls
or retains the previous attention module as a registered child.
"""

from __future__ import annotations
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
import importlib
import inspect
import json
from pathlib import Path
from collections.abc import Callable, Iterator, Mapping
import torch
from torch import Tensor, nn
from ..backbones.spec import ModelGeometry
from ..experiment_contract import (
    HEAD_DIM,
    HIDDEN_SIZE,
    NUM_KEY_VALUE_HEADS,
    NUM_LAYERS,
    NUM_QUERY_HEADS,
)
from ..package_resources import qwen2_attention_compatibility_fixture_path

_DEFAULT_COMPATIBILITY_PATH = qwen2_attention_compatibility_fixture_path()
_EXPECTED_TRANSFORMERS_VERSION = "4.51.3"
_EXPECTED_ROPE_MODULE = "transformers.models.qwen2.modeling_qwen2"
_EXPECTED_ROPE_SIGNATURE = "(q, k, cos, sin, position_ids=None, unsqueeze_dim=1)"
_EXPECTED_ATTENTION_FORWARD_SIGNATURE = "(hidden_states: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor], attention_mask: Optional[torch.Tensor], past_key_value: Optional[transformers.cache_utils.Cache] = None, cache_position: Optional[torch.LongTensor] = None, **kwargs: typing_extensions.Unpack[transformers.modeling_flash_attention_utils.FlashAttentionKwargs]) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]"


class AttentionContractError(RuntimeError):
    """Raised when a measured Qwen contract or pilot input is violated."""


_UNSET_ATTENTION_MASK = object()
_UNAVAILABLE_ATTENTION_MASK_VERSION = object()


@dataclass
class _AttentionValidationSession:
    """One context-local, outer-model-forward validation scope."""

    diagnostic: bool
    attention_mask: Tensor | None | object = field(default=_UNSET_ATTENTION_MASK)
    attention_mask_version: int | None | object = field(default=_UNSET_ATTENTION_MASK)


_ATTENTION_VALIDATION_SESSION: ContextVar[_AttentionValidationSession | None] = (
    ContextVar("qwen_lora_attention_validation_session", default=None)
)


@contextmanager
def attention_validation_session(*, diagnostic: bool = False) -> Iterator[None]:
    """Scope attention validation to one outer model forward.

    Every entry receives a new context-local session.  A nested scope is
    therefore independent of its parent and always restores that parent with
    its ContextVar token, including when the nested forward raises.

    In a normal session, callers must treat the shared attention mask as an
    immutable input.  The mutation counter detects ordinary in-place writes,
    but ``Tensor.data`` is an unsupported PyTorch escape hatch and is not
    guaranteed to update that counter.  Detecting such writes would require a
    full ``[N, N]`` comparison at every adapter, which this scoped validation
    deliberately avoids.
    """
    if type(diagnostic) is not bool:
        raise TypeError("diagnostic must be a boolean")
    token = _ATTENTION_VALIDATION_SESSION.set(
        _AttentionValidationSession(diagnostic=diagnostic)
    )
    try:
        yield
    finally:
        _ATTENTION_VALIDATION_SESSION.reset(token)


@contextmanager
def _attention_validation_replay_session(
    snapshot: _AttentionValidationSession,
) -> Iterator[None]:
    """Install one checkpoint frame's captured validation state for replay."""
    token = _ATTENTION_VALIDATION_SESSION.set(snapshot)
    try:
        yield
    finally:
        _ATTENTION_VALIDATION_SESSION.reset(token)


def attention_validation_checkpoint_context_fn() -> (
    tuple[AbstractContextManager[None], AbstractContextManager[None]]
):
    """Capture validation state once for a non-reentrant checkpoint frame."""
    active = _ATTENTION_VALIDATION_SESSION.get()
    if active is None:
        raise AttentionContractError(
            "checkpoint context creation requires an active attention-validation session"
        )
    snapshot = _AttentionValidationSession(
        diagnostic=active.diagnostic,
        attention_mask=active.attention_mask,
        attention_mask_version=active.attention_mask_version,
    )
    return (nullcontext(), _attention_validation_replay_session(snapshot))


def run_attention_validation_forward(
    forward: Callable[..., object], /, *args: object, **kwargs: object
) -> object:
    """Run one production model-forward callable in a fresh normal session."""
    if not callable(forward):
        raise TypeError("forward must be callable")
    with attention_validation_session():
        return forward(*args, **kwargs)


def attention_validation_requires_full_value_audit() -> bool:
    """Whether this call must scan every common-adapter intermediate value."""
    active = _ATTENTION_VALIDATION_SESSION.get()
    return active is None or active.diagnostic


def is_checkpoint_early_stop_exception(exc: BaseException) -> bool:
    """Return whether ``exc`` is PyTorch non-reentrant checkpoint control flow.

    The private exception is deliberately identified structurally: importing
    it would couple the experiment to a private PyTorch symbol, while wrapping
    it breaks checkpoint's own saved-tensor accounting.
    """
    return (
        type(exc).__module__ == "torch.utils.checkpoint"
        and type(exc).__name__ == "_StopRecomputationError"
    )


@dataclass(frozen=True)
class Qwen2AttentionContract:
    """The immutable portion of the server-collected compatibility evidence."""

    transformers_version: str
    attention_module: str
    attention_class_name: str
    attention_forward_signature: str
    attention_forward_return_arity: int
    rope_ownership: str
    rotary_module: str
    rotary_signature: str
    q_projection_shape: tuple[int, int]
    k_projection_shape: tuple[int, int]
    v_projection_shape: tuple[int, int]
    o_projection_shape: tuple[int, int]


@dataclass(frozen=True)
class _AttentionTopology:
    layer_id: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int


@dataclass(frozen=True)
class AttentionFamilyContract:
    """The projection details that genuinely differ between model families."""

    family: str
    qkv_bias: bool
    output_bias: bool


QWEN_GEOMETRY = ModelGeometry(
    NUM_LAYERS, HIDDEN_SIZE, NUM_QUERY_HEADS, NUM_KEY_VALUE_HEADS, HEAD_DIM
)
QWEN_FAMILY_CONTRACT = AttentionFamilyContract(
    family="qwen2", qkv_bias=True, output_bias=False
)


def load_qwen2_attention_contract(path: str | Path) -> Qwen2AttentionContract:
    """Load and rigorously validate the measured Qwen compatibility fixture.

    The fixture is not a loose hint: unexpected fields, omitted evidence, or
    an environment different from the approved Qwen2.5-0.5B geometry are all
    rejected before a replacement attention module can be constructed.
    """
    fixture_path = Path(path)
    try:
        raw = fixture_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AttentionContractError(
            f"could not read Qwen attention compatibility fixture: {fixture_path}"
        ) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AttentionContractError(
            f"Qwen attention compatibility fixture is not valid JSON: {fixture_path}"
        ) from exc
    from ..operator_imports import normalize_operator_identity

    if isinstance(payload, Mapping):
        payload = normalize_operator_identity(payload)
    root = _mapping_with_exact_keys(
        payload,
        "fixture",
        {
            "apply_rotary_pos_emb",
            "collected_at",
            "command",
            "cpu_forward",
            "oal_attention",
            "model_config_sha256",
            "model_path",
            "position_embeddings",
            "qwen2_attention",
            "qwen2_config",
            "rope_contract",
            "schema_version",
            "transformers",
        },
    )
    _require_exact_int(root["schema_version"], "fixture.schema_version", 1)
    _require_nonempty_string(root["collected_at"], "fixture.collected_at")
    _validate_command(root["command"])
    _require_nonempty_string(root["model_path"], "fixture.model_path")
    _validate_sha256(root["model_config_sha256"], "fixture.model_config_sha256")
    _validate_operator_evidence(root["oal_attention"])
    transformers = _mapping_with_exact_keys(
        root["transformers"], "fixture.transformers", {"path", "version"}
    )
    _require_nonempty_string(transformers["path"], "fixture.transformers.path")
    _require_exact_string(
        transformers["version"],
        "fixture.transformers.version",
        _EXPECTED_TRANSFORMERS_VERSION,
    )
    _validate_qwen_config(root["qwen2_config"])
    attention = _validate_qwen_attention(root["qwen2_attention"])
    rope_contract = _mapping_with_exact_keys(
        root["rope_contract"],
        "fixture.rope_contract",
        {"attention_forward_return_arity", "ownership"},
    )
    _require_exact_int(
        rope_contract["attention_forward_return_arity"],
        "fixture.rope_contract.attention_forward_return_arity",
        2,
    )
    _require_exact_string(
        rope_contract["ownership"], "fixture.rope_contract.ownership", "model"
    )
    rotary = _validate_rotary_evidence(root["apply_rotary_pos_emb"])
    _validate_cpu_forward(root["cpu_forward"])
    _validate_fixture_position_embeddings(root["position_embeddings"])
    return Qwen2AttentionContract(
        transformers_version=transformers["version"],
        attention_module=attention["module"],
        attention_class_name=attention["class_name"],
        attention_forward_signature=attention["forward_signature"],
        attention_forward_return_arity=rope_contract["attention_forward_return_arity"],
        rope_ownership=rope_contract["ownership"],
        rotary_module=rotary["module"],
        rotary_signature=rotary["signature"],
        q_projection_shape=(HIDDEN_SIZE, HIDDEN_SIZE),
        k_projection_shape=(NUM_KEY_VALUE_HEADS * HEAD_DIM, HIDDEN_SIZE),
        v_projection_shape=(NUM_KEY_VALUE_HEADS * HEAD_DIM, HIDDEN_SIZE),
        o_projection_shape=(HIDDEN_SIZE, HIDDEN_SIZE),
    )


class CommonAttentionAdapter(nn.Module):
    """A no-cache family-neutral shell around one method head callable."""

    def __init__(
        self,
        original_attention: nn.Module,
        *,
        layer_id: int,
        geometry: ModelGeometry,
        family_contract: AttentionFamilyContract,
        apply_rotary_pos_emb: Callable[[Tensor, Tensor, Tensor, Tensor], object],
        method_callable: Callable[[Tensor, Tensor, Tensor, int], Tensor],
    ) -> None:
        super().__init__()
        topology = _build_topology(layer_id, geometry)
        projections = _extract_and_validate_family_projections(
            original_attention, geometry, family_contract
        )
        if not callable(apply_rotary_pos_emb):
            raise AttentionContractError("apply_rotary_pos_emb must be callable")
        if not callable(method_callable):
            raise AttentionContractError("method_callable must be callable")
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = projections
        object.__setattr__(self, "_topology", topology)
        object.__setattr__(self, "_geometry", geometry)
        object.__setattr__(self, "_family_contract", family_contract)
        object.__setattr__(self, "_apply_rotary_pos_emb", apply_rotary_pos_emb)
        object.__setattr__(self, "_method_callable", method_callable)

    @property
    def layer_id(self) -> int:
        return self._topology.layer_id

    @property
    def num_attention_heads(self) -> int:
        return self._topology.num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        return self._topology.num_key_value_heads

    @property
    def head_dim(self) -> int:
        return self._topology.head_dim

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
        past_key_value: object | None = None,
        cache_position: Tensor | None = None,
        **kwargs: object,
    ) -> tuple[Tensor, None]:
        batch_size, sequence_length = _validate_hidden_states(
            hidden_states, geometry=self._geometry
        )
        validate_no_cache_and_exact_causal_mask(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            kwargs=kwargs,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        q, k, v = project_qkv_and_apply_rope(
            q_proj=self.q_proj,
            k_proj=self.k_proj,
            v_proj=self.v_proj,
            apply_rotary_pos_emb=self._apply_rotary_pos_emb,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            layer_id=self.layer_id,
            geometry=self._geometry,
        )
        head_output = self._method_callable(q, k, v, self.layer_id)
        _validate_head_output(head_output, q)
        output = restore_and_project_output(
            head_output=head_output,
            o_proj=self.o_proj,
            batch_size=batch_size,
            sequence_length=sequence_length,
            layer_id=self.layer_id,
            geometry=self._geometry,
        )
        return (output, None)


class QwenAttentionAdapter(CommonAttentionAdapter):
    """Legacy Qwen entry over the family-neutral attention shell.

    ``head_attention`` is the only method-specific operation.  Subclasses must
    return a finite tensor matching the RoPE-rotated query tensor.
    """

    def __init__(
        self,
        original_attention: nn.Module,
        *,
        layer_id: int,
        compatibility_path: str | Path = _DEFAULT_COMPATIBILITY_PATH,
        apply_rotary_pos_emb: (
            Callable[[Tensor, Tensor, Tensor, Tensor], object] | None
        ) = None,
        geometry: ModelGeometry = QWEN_GEOMETRY,
        family_contract: AttentionFamilyContract = QWEN_FAMILY_CONTRACT,
    ) -> None:
        contract = (
            load_qwen2_attention_contract(compatibility_path)
            if family_contract.family == "qwen2"
            else None
        )
        rotary = apply_rotary_pos_emb
        if rotary is None:
            if contract is None:
                raise AttentionContractError(
                    "non-Qwen attention requires an explicit family RoPE callable"
                )
            rotary = _load_default_rotary(contract)
        super().__init__(
            original_attention,
            layer_id=layer_id,
            geometry=geometry,
            family_contract=family_contract,
            apply_rotary_pos_emb=rotary,
            method_callable=self.head_attention,
        )
        object.__setattr__(self, "_contract", contract)

    def head_attention(self, q: Tensor, k: Tensor, v: Tensor, layer_id: int) -> Tensor:
        """Return method-specific heads; subclasses must implement this."""
        del q, k, v, layer_id
        raise NotImplementedError(
            "QwenAttentionAdapter subclasses must implement head_attention"
        )


def validate_no_cache_and_exact_causal_mask(
    *,
    hidden_states: Tensor,
    attention_mask: Tensor | None,
    past_key_value: object | None,
    cache_position: Tensor | None,
    kwargs: Mapping[str, object],
    batch_size: int,
    sequence_length: int,
) -> None:
    """Reject cache, padding, windows, and other unsupported attention modes."""
    if past_key_value is not None:
        raise AttentionContractError(
            "past_key_value is unsupported by the no-cache pilot"
        )
    for name, value in kwargs.items():
        if name == "position_ids":
            _validate_position_ids(
                value,
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=hidden_states.device,
            )
        elif name in {"use_cache", "output_attentions"}:
            if value is not None and value is not False:
                raise AttentionContractError(f"{name}=True is unsupported by the pilot")
        elif value is not None and value is not False:
            raise AttentionContractError(
                f"unsupported meaningful attention kwarg: {name}"
            )
    validate_cache_position(
        cache_position, sequence_length=sequence_length, device=hidden_states.device
    )
    _validate_attention_mask_for_current_session(
        attention_mask,
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=hidden_states.device,
    )


def _validate_position_ids(
    position_ids: object, *, batch_size: int, sequence_length: int, device: torch.device
) -> None:
    """Validate Qwen's redundant position IDs against model-owned RoPE inputs."""
    if position_ids is None:
        return
    if not isinstance(position_ids, Tensor):
        raise AttentionContractError("position_ids must be None or a torch.Tensor")
    if position_ids.device != device:
        raise AttentionContractError("position_ids device must match hidden_states")
    if position_ids.dtype is not torch.int64:
        raise AttentionContractError("position_ids must use torch.int64")
    if position_ids.ndim != 2 or position_ids.shape[1] != sequence_length:
        raise AttentionContractError(
            "position_ids must have shape [1, N] or [B, N] for the current sequence"
        )
    if position_ids.shape[0] not in (1, batch_size):
        raise AttentionContractError(
            "position_ids must have one row or one row per input batch item"
        )
    expected = torch.arange(sequence_length, device=device, dtype=torch.int64).expand(
        position_ids.shape[0], -1
    )
    if not torch.equal(position_ids, expected):
        raise AttentionContractError(
            "position_ids must equal torch.arange(sequence_length) for the no-cache pilot"
        )


def validate_cache_position(
    cache_position: Tensor | None, *, sequence_length: int, device: torch.device
) -> None:
    """Accept only the fresh-token ``arange(N)`` cache-position convention."""
    if cache_position is None:
        return
    if not isinstance(cache_position, Tensor):
        raise AttentionContractError("cache_position must be None or a torch.Tensor")
    if cache_position.device != device:
        raise AttentionContractError("cache_position device must match hidden_states")
    if cache_position.dtype is not torch.int64:
        raise AttentionContractError("cache_position must use torch.int64")
    if cache_position.shape != (sequence_length,):
        raise AttentionContractError(
            f"cache_position must have shape ({sequence_length},), got {tuple(cache_position.shape)}"
        )
    expected = torch.arange(sequence_length, device=device, dtype=torch.int64)
    if not torch.equal(cache_position, expected):
        raise AttentionContractError(
            "cache_position must equal torch.arange(sequence_length)"
        )


def project_qkv_and_apply_rope(
    *,
    q_proj: nn.Module,
    k_proj: nn.Module,
    v_proj: nn.Module,
    apply_rotary_pos_emb: Callable[[Tensor, Tensor, Tensor, Tensor], object],
    hidden_states: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    layer_id: int,
    geometry: ModelGeometry = QWEN_GEOMETRY,
) -> tuple[Tensor, Tensor, Tensor]:
    """Project Q/K/V, reshape native heads, and rotate Q/K only."""
    batch_size, sequence_length = _validate_hidden_states(
        hidden_states, geometry=geometry
    )
    cos, sin = _validate_position_embeddings(
        position_embeddings,
        batch_size=batch_size,
        sequence_length=sequence_length,
        hidden_states=hidden_states,
        head_dim=geometry.head_dim,
    )
    q_flat = _validate_projection_output(
        "q_proj",
        _call_projection("q_proj", q_proj, hidden_states, layer_id=layer_id),
        (batch_size, sequence_length, geometry.hidden_size),
        hidden_states,
    )
    k_flat = _validate_projection_output(
        "k_proj",
        _call_projection("k_proj", k_proj, hidden_states, layer_id=layer_id),
        (batch_size, sequence_length, geometry.num_kv_heads * geometry.head_dim),
        hidden_states,
    )
    v_flat = _validate_projection_output(
        "v_proj",
        _call_projection("v_proj", v_proj, hidden_states, layer_id=layer_id),
        (batch_size, sequence_length, geometry.num_kv_heads * geometry.head_dim),
        hidden_states,
    )
    q = q_flat.reshape(
        batch_size, sequence_length, geometry.num_query_heads, geometry.head_dim
    ).transpose(1, 2)
    k = k_flat.reshape(
        batch_size, sequence_length, geometry.num_kv_heads, geometry.head_dim
    ).transpose(1, 2)
    v = v_flat.reshape(
        batch_size, sequence_length, geometry.num_kv_heads, geometry.head_dim
    ).transpose(1, 2)
    try:
        rotated = apply_rotary_pos_emb(q, k, cos, sin)
    except Exception as exc:
        if is_checkpoint_early_stop_exception(exc):
            raise
        raise AttentionContractError("apply_rotary_pos_emb failed") from exc
    if not isinstance(rotated, tuple) or len(rotated) != 2:
        raise AttentionContractError(
            "apply_rotary_pos_emb must return a two-tensor tuple"
        )
    rotated_q, rotated_k = rotated
    _validate_rotary_output("RoPE q", rotated_q, q)
    _validate_rotary_output("RoPE k", rotated_k, k)
    return (rotated_q.contiguous(), rotated_k.contiguous(), v.contiguous())


def restore_and_project_output(
    *,
    head_output: Tensor,
    o_proj: nn.Module,
    batch_size: int,
    sequence_length: int,
    layer_id: int,
    geometry: ModelGeometry = QWEN_GEOMETRY,
) -> Tensor:
    """Restore the native hidden layout and invoke ``o_proj`` exactly once."""
    restored = (
        head_output.transpose(1, 2)
        .contiguous()
        .reshape(batch_size, sequence_length, geometry.hidden_size)
    )
    output = _call_projection("o_proj", o_proj, restored, layer_id=layer_id)
    return _validate_projection_output(
        "o_proj",
        output,
        (batch_size, sequence_length, geometry.hidden_size),
        head_output,
    )


def _build_topology(
    layer_id: int, geometry: ModelGeometry = QWEN_GEOMETRY
) -> _AttentionTopology:
    if isinstance(layer_id, bool) or not isinstance(layer_id, int):
        raise AttentionContractError("layer_id must be an integer")
    if not 0 <= layer_id < geometry.num_layers:
        raise AttentionContractError(
            f"layer_id must be in [0, {geometry.num_layers}), got {layer_id}"
        )
    return _AttentionTopology(
        layer_id=layer_id,
        num_attention_heads=geometry.num_query_heads,
        num_key_value_heads=geometry.num_kv_heads,
        head_dim=geometry.head_dim,
    )


def _extract_and_validate_projections(
    original_attention: nn.Module, contract: Qwen2AttentionContract
) -> tuple[nn.Linear, nn.Linear, nn.Linear, nn.Linear]:
    return _extract_and_validate_family_projections(
        original_attention, QWEN_GEOMETRY, QWEN_FAMILY_CONTRACT
    )


def _extract_and_validate_family_projections(
    original_attention: nn.Module,
    geometry: ModelGeometry,
    family_contract: AttentionFamilyContract,
) -> tuple[nn.Linear, nn.Linear, nn.Linear, nn.Linear]:
    if not isinstance(original_attention, nn.Module):
        raise AttentionContractError("original_attention must be an nn.Module")
    if getattr(original_attention, "head_dim", None) != geometry.head_dim:
        raise AttentionContractError(
            f"original_attention.head_dim must equal {geometry.head_dim}"
        )
    projections: list[nn.Linear] = []
    expected = (
        (
            "q_proj",
            (geometry.hidden_size, geometry.hidden_size),
            family_contract.qkv_bias,
        ),
        (
            "k_proj",
            (geometry.num_kv_heads * geometry.head_dim, geometry.hidden_size),
            family_contract.qkv_bias,
        ),
        (
            "v_proj",
            (geometry.num_kv_heads * geometry.head_dim, geometry.hidden_size),
            family_contract.qkv_bias,
        ),
        (
            "o_proj",
            (geometry.hidden_size, geometry.hidden_size),
            family_contract.output_bias,
        ),
    )
    for name, shape, requires_bias in expected:
        projection = getattr(original_attention, name, None)
        if not isinstance(projection, nn.Linear):
            raise AttentionContractError(f"original_attention.{name} must be nn.Linear")
        if tuple(projection.weight.shape) != shape:
            raise AttentionContractError(
                f"original_attention.{name}.weight must have shape {shape}"
            )
        if (projection.bias is not None) != requires_bias:
            raise AttentionContractError(
                f"original_attention.{name} bias does not match {family_contract.family}"
            )
        projections.append(projection)
    if len({id(projection) for projection in projections}) != len(projections):
        raise AttentionContractError(
            "attention projections must be four distinct module instances"
        )
    return (projections[0], projections[1], projections[2], projections[3])


def _load_default_rotary(
    contract: Qwen2AttentionContract,
) -> Callable[[Tensor, Tensor, Tensor, Tensor], object]:
    try:
        transformers = importlib.import_module("transformers")
        rotary_module = importlib.import_module(contract.rotary_module)
    except (ImportError, ModuleNotFoundError) as exc:
        raise AttentionContractError(
            "could not import the fixture-pinned Transformers Qwen RoPE implementation"
        ) from exc
    version = getattr(transformers, "__version__", None)
    if version != contract.transformers_version:
        raise AttentionContractError(
            f"Transformers version does not match the compatibility fixture: expected {contract.transformers_version}, got {version!r}"
        )
    rotary = getattr(rotary_module, "apply_rotary_pos_emb", None)
    if not callable(rotary):
        raise AttentionContractError(
            f"fixture module {contract.rotary_module!r} has no callable apply_rotary_pos_emb"
        )
    try:
        signature = str(inspect.signature(rotary))
    except (TypeError, ValueError) as exc:
        raise AttentionContractError(
            "could not inspect apply_rotary_pos_emb signature"
        ) from exc
    if signature != contract.rotary_signature:
        raise AttentionContractError(
            f"apply_rotary_pos_emb signature does not match the compatibility fixture: expected {contract.rotary_signature!r}, got {signature!r}"
        )
    return rotary


def _validate_hidden_states(
    hidden_states: Tensor, *, geometry: ModelGeometry = QWEN_GEOMETRY
) -> tuple[int, int]:
    if not isinstance(hidden_states, Tensor):
        raise AttentionContractError("hidden_states must be a torch.Tensor")
    if hidden_states.ndim != 3 or hidden_states.shape[-1] != geometry.hidden_size:
        raise AttentionContractError(
            f"hidden_states must have shape [B, N, {geometry.hidden_size}], got {tuple(hidden_states.shape)}"
        )
    if hidden_states.shape[0] <= 0 or hidden_states.shape[1] <= 0:
        raise AttentionContractError(
            "hidden_states batch and sequence dimensions must be positive"
        )
    if not hidden_states.is_floating_point():
        raise AttentionContractError("hidden_states must have a floating dtype")
    if not hidden_states.is_contiguous():
        raise AttentionContractError("hidden_states must be contiguous")
    _require_finite_tensor(hidden_states, "hidden_states")
    return (int(hidden_states.shape[0]), int(hidden_states.shape[1]))


def _validate_position_embeddings(
    position_embeddings: tuple[Tensor, Tensor],
    *,
    batch_size: int,
    sequence_length: int,
    hidden_states: Tensor,
    head_dim: int = HEAD_DIM,
) -> tuple[Tensor, Tensor]:
    if not isinstance(position_embeddings, tuple) or len(position_embeddings) != 2:
        raise AttentionContractError("position_embeddings must be a two-tensor tuple")
    cos, sin = position_embeddings
    expected_shape = (batch_size, sequence_length, head_dim)
    for name, tensor in (("cos", cos), ("sin", sin)):
        if not isinstance(tensor, Tensor):
            raise AttentionContractError(
                f"position_embeddings {name} must be a torch.Tensor"
            )
        if tuple(tensor.shape) != expected_shape:
            raise AttentionContractError(
                f"position_embeddings {name} must have shape {expected_shape}"
            )
        if tensor.device != hidden_states.device:
            raise AttentionContractError(
                f"position_embeddings {name} device must match hidden_states"
            )
        if tensor.dtype != hidden_states.dtype:
            raise AttentionContractError(
                f"position_embeddings {name} dtype must match hidden_states"
            )
        if not tensor.is_contiguous():
            raise AttentionContractError(
                f"position_embeddings {name} must be contiguous"
            )
        _require_finite_tensor(tensor, f"position_embeddings {name}")
    return (cos, sin)


def _validate_attention_mask(
    attention_mask: Tensor | None,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> None:
    _validate_attention_mask_structure(
        attention_mask,
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=device,
    )
    if attention_mask is None:
        return
    if attention_mask.ndim == 2:
        if attention_mask.dtype is torch.bool:
            if not bool(torch.all(attention_mask).item()):
                raise AttentionContractError(
                    "attention_mask 2D bool mask must be all valid"
                )
            return
        if not bool(torch.all(attention_mask == 1).item()):
            raise AttentionContractError(
                "attention_mask 2D numeric mask must be all-one"
            )
        return
    expected_bool = (
        torch.tril(
            torch.ones(
                (sequence_length, sequence_length), device=device, dtype=torch.bool
            )
        )
        .reshape(1, 1, sequence_length, sequence_length)
        .expand(batch_size, -1, -1, -1)
    )
    if attention_mask.dtype is torch.bool:
        if not torch.equal(attention_mask, expected_bool):
            raise AttentionContractError(
                "attention_mask 4D bool mask must be exact causal lower triangle"
            )
        return
    lower = attention_mask.masked_select(expected_bool)
    upper = attention_mask.masked_select(~expected_bool)
    if not bool(torch.all(lower == 0).item()):
        raise AttentionContractError(
            "attention_mask additive lower triangle must be zero"
        )
    finite_min = torch.finfo(attention_mask.dtype).min
    has_negative_infinity = bool(torch.all(torch.isneginf(upper)).item())
    has_dtype_minimum = bool(torch.all(upper == finite_min).item())
    if not (has_negative_infinity or has_dtype_minimum):
        raise AttentionContractError(
            "attention_mask additive upper triangle must use consistent -inf or dtype minimum"
        )


def _validate_attention_mask_for_current_session(
    attention_mask: Tensor | None,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> None:
    """Audit a mask fully once per normal outer forward, then by identity."""
    session = _ATTENTION_VALIDATION_SESSION.get()
    if session is None or session.diagnostic:
        _validate_attention_mask(
            attention_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=device,
        )
        return
    if session.attention_mask is _UNSET_ATTENTION_MASK:
        _validate_attention_mask(
            attention_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=device,
        )
        session.attention_mask = attention_mask
        session.attention_mask_version = _attention_mask_version(attention_mask)
        return
    if attention_mask is not session.attention_mask:
        raise AttentionContractError(
            "all custom attention adapters in one model forward must receive the exact same attention_mask object"
        )
    if _attention_mask_version(attention_mask) != session.attention_mask_version:
        raise AttentionContractError(
            "attention_mask version changed during one model forward; refusing to reuse it"
        )
    _validate_attention_mask_structure(
        attention_mask,
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=device,
    )


def _validate_attention_mask_structure(
    attention_mask: Tensor | None,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> None:
    """Keep the non-allocating dynamic mask contract on every adapter call."""
    if attention_mask is None:
        return
    if not isinstance(attention_mask, Tensor):
        raise AttentionContractError("attention_mask must be None or a torch.Tensor")
    if attention_mask.device != device or attention_mask.device.type == "meta":
        raise AttentionContractError("attention_mask device must match hidden_states")
    if attention_mask.ndim == 2:
        if tuple(attention_mask.shape) != (batch_size, sequence_length):
            raise AttentionContractError("attention_mask 2D shape must equal [B, N]")
        if attention_mask.dtype is torch.bool:
            return
        if (
            attention_mask.is_floating_point()
            or attention_mask.dtype in _INTEGER_DTYPES
        ):
            return
        raise AttentionContractError(
            "attention_mask 2D mask must be bool or numeric all-one"
        )
    if attention_mask.ndim != 4 or tuple(attention_mask.shape) != (
        batch_size,
        1,
        sequence_length,
        sequence_length,
    ):
        raise AttentionContractError(
            "attention_mask must be full-valid 2D or exact causal 4D"
        )
    if attention_mask.dtype is torch.bool or attention_mask.is_floating_point():
        return
    raise AttentionContractError(
        "attention_mask 4D mask must be bool or floating additive"
    )


def _attention_mask_version(attention_mask: Tensor | None) -> int | None | object:
    """Return a tracked mutation counter, or a private unavailable sentinel.

    PyTorch inference tensors intentionally do not expose ``Tensor._version``.
    They remain safe here because the session still enforces object identity
    and structural checks, while a fresh outer-forward session fully audits
    the mask again.
    """
    if attention_mask is None:
        return None
    try:
        version = attention_mask._version
    except RuntimeError:
        return _UNAVAILABLE_ATTENTION_MASK_VERSION
    if type(version) is not int:
        return _UNAVAILABLE_ATTENTION_MASK_VERSION
    return version


def _validate_projection_output(
    name: str, value: object, expected_shape: tuple[int, int, int], reference: Tensor
) -> Tensor:
    if not isinstance(value, Tensor):
        raise AttentionContractError(f"{name} must return a torch.Tensor")
    if tuple(value.shape) != expected_shape:
        raise AttentionContractError(f"{name} must return shape {expected_shape}")
    if value.device != reference.device or value.dtype != reference.dtype:
        raise AttentionContractError(
            f"{name} output device and dtype must match hidden_states"
        )
    if not value.is_contiguous():
        raise AttentionContractError(f"{name} output must be contiguous")
    _require_finite_tensor(value, name)
    return value


def _call_projection(
    name: str, projection: nn.Module, input_tensor: Tensor, *, layer_id: int
) -> object:
    """Call one Qwen projection with actionable device/dtype diagnostics."""
    try:
        return projection(input_tensor)
    except Exception as exc:
        if is_checkpoint_early_stop_exception(exc):
            raise
        weight = getattr(projection, "weight", None)
        if isinstance(weight, Tensor):
            weight_context = f"weight(device={weight.device}, dtype={weight.dtype})"
        else:
            weight_context = "weight(device=<unavailable>, dtype=<unavailable>)"
        raise AttentionContractError(
            f"projection call failed: layer_id={layer_id}, projection={name}, input(device={input_tensor.device}, dtype={input_tensor.dtype}), {weight_context}"
        ) from exc


def _validate_rotary_output(name: str, value: object, reference: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise AttentionContractError(f"{name} must be a torch.Tensor")
    if value.shape != reference.shape:
        raise AttentionContractError(f"{name} has invalid shape {tuple(value.shape)}")
    if value.device != reference.device or value.dtype != reference.dtype:
        raise AttentionContractError(
            f"{name} device and dtype must match projected tensor"
        )
    _require_finite_tensor(value, name)


def _validate_head_output(head_output: object, q: Tensor) -> None:
    if not isinstance(head_output, Tensor):
        raise AttentionContractError("head_attention must return a torch.Tensor")
    if head_output.shape != q.shape:
        raise AttentionContractError(
            f"head_attention must return shape {tuple(q.shape)}, got {tuple(head_output.shape)}"
        )
    if head_output.device != q.device or head_output.dtype != q.dtype:
        raise AttentionContractError(
            "head_attention output device and dtype must match q"
        )
    if not head_output.is_contiguous():
        raise AttentionContractError("head_attention output must be contiguous")
    _require_finite_tensor(head_output, "head_attention")


def _require_finite_tensor(value: Tensor, name: str) -> None:
    if value.device.type == "meta":
        raise AttentionContractError(f"{name} cannot be a meta tensor")
    if not value.is_floating_point():
        raise AttentionContractError(f"{name} must contain only finite floating values")
    if attention_validation_requires_full_value_audit() and (
        not bool(torch.isfinite(value).all().item())
    ):
        raise AttentionContractError(f"{name} must contain only finite floating values")


def _mapping_with_exact_keys(
    value: object, name: str, expected_keys: set[str]
) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise AttentionContractError(f"{name} must be an object")
    actual_keys = set(value)
    unknown = actual_keys - expected_keys
    missing = expected_keys - actual_keys
    if unknown:
        qualifier = "top-level " if name == "fixture" else ""
        raise AttentionContractError(
            f"{name} has unknown {qualifier}field(s): {', '.join(sorted(unknown))}"
        )
    if missing:
        raise AttentionContractError(
            f"{name} is missing field(s): {', '.join(sorted(missing))}"
        )
    return value


def _require_nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AttentionContractError(f"{name} must be a non-empty string")
    return value


def _require_exact_string(value: object, name: str, expected: str) -> str:
    actual = _require_nonempty_string(value, name)
    if actual != expected:
        raise AttentionContractError(f"{name} must equal {expected!r}, got {actual!r}")
    return actual


def _require_exact_int(value: object, name: str, expected: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise AttentionContractError(f"{name} must equal integer {expected}")
    return value


def _validate_command(value: object) -> None:
    if (
        not isinstance(value, list)
        or not value
        or (not all((isinstance(item, str) and item for item in value)))
    ):
        raise AttentionContractError("fixture.command must be a non-empty string list")
    if not value[0].endswith("collect_qwen2_compatibility.py"):
        raise AttentionContractError(
            "fixture.command must record the compatibility collector"
        )


def _validate_sha256(value: object, name: str) -> None:
    text = _require_nonempty_string(value, name)
    if len(text) != 64 or any(
        (character not in "0123456789abcdef" for character in text)
    ):
        raise AttentionContractError(f"{name} must be a lowercase SHA-256 digest")


def _validate_operator_evidence(value: object) -> None:
    evidence = _mapping_with_exact_keys(
        value, "fixture.oal_attention", {"available", "path", "project_path", "version"}
    )
    if evidence["available"] is not True:
        raise AttentionContractError("fixture.oal_attention.available must be true")
    for name in ("path", "project_path", "version"):
        _require_nonempty_string(evidence[name], f"fixture.oal_attention.{name}")


def _validate_qwen_config(value: object) -> None:
    config = _mapping_with_exact_keys(
        value,
        "fixture.qwen2_config",
        {
            "class_name",
            "head_dim",
            "hidden_size",
            "model_type",
            "module",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
        },
    )
    _require_exact_string(
        config["class_name"], "fixture.qwen2_config.class_name", "Qwen2Config"
    )
    _require_exact_string(
        config["model_type"], "fixture.qwen2_config.model_type", "qwen2"
    )
    _require_exact_string(
        config["module"],
        "fixture.qwen2_config.module",
        "transformers.models.qwen2.configuration_qwen2",
    )
    for name, expected in (
        ("head_dim", HEAD_DIM),
        ("hidden_size", HIDDEN_SIZE),
        ("num_attention_heads", NUM_QUERY_HEADS),
        ("num_hidden_layers", NUM_LAYERS),
        ("num_key_value_heads", NUM_KEY_VALUE_HEADS),
    ):
        _require_exact_int(config[name], f"fixture.qwen2_config.{name}", expected)


def _validate_qwen_attention(value: object) -> Mapping[str, object]:
    attention = _mapping_with_exact_keys(
        value,
        "fixture.qwen2_attention",
        {"attributes", "class_name", "forward_signature", "init_signature", "module"},
    )
    _require_exact_string(
        attention["class_name"], "fixture.qwen2_attention.class_name", "Qwen2Attention"
    )
    _require_exact_string(
        attention["module"], "fixture.qwen2_attention.module", _EXPECTED_ROPE_MODULE
    )
    _require_exact_string(
        attention["forward_signature"],
        "fixture.qwen2_attention.forward_signature",
        _EXPECTED_ATTENTION_FORWARD_SIGNATURE,
    )
    _require_exact_string(
        attention["init_signature"],
        "fixture.qwen2_attention.init_signature",
        "(config: transformers.models.qwen2.configuration_qwen2.Qwen2Config, layer_idx: int)",
    )
    attributes = _mapping_with_exact_keys(
        attention["attributes"],
        "fixture.qwen2_attention.attributes",
        {"head_dim", "k_proj", "o_proj", "q_proj", "v_proj"},
    )
    head_dim = _mapping_with_exact_keys(
        attributes["head_dim"],
        "fixture.qwen2_attention.attributes.head_dim",
        {"type", "value"},
    )
    _require_exact_string(
        head_dim["type"], "fixture.qwen2_attention.attributes.head_dim.type", "int"
    )
    _require_exact_int(
        head_dim["value"], "fixture.qwen2_attention.attributes.head_dim.value", HEAD_DIM
    )
    for name, shape, bias_shape in (
        ("q_proj", [HIDDEN_SIZE, HIDDEN_SIZE], [HIDDEN_SIZE]),
        (
            "k_proj",
            [NUM_KEY_VALUE_HEADS * HEAD_DIM, HIDDEN_SIZE],
            [NUM_KEY_VALUE_HEADS * HEAD_DIM],
        ),
        (
            "v_proj",
            [NUM_KEY_VALUE_HEADS * HEAD_DIM, HIDDEN_SIZE],
            [NUM_KEY_VALUE_HEADS * HEAD_DIM],
        ),
        ("o_proj", [HIDDEN_SIZE, HIDDEN_SIZE], None),
    ):
        projection = _mapping_with_exact_keys(
            attributes[name],
            f"fixture.qwen2_attention.attributes.{name}",
            {"bias_shape", "class_name", "module", "weight_shape"},
        )
        _require_exact_string(
            projection["class_name"],
            f"fixture.qwen2_attention.attributes.{name}.class_name",
            "Linear",
        )
        _require_exact_string(
            projection["module"],
            f"fixture.qwen2_attention.attributes.{name}.module",
            "torch.nn.modules.linear",
        )
        _require_exact_list(
            projection["weight_shape"],
            f"fixture.qwen2_attention.attributes.{name}.weight_shape",
            shape,
        )
        if bias_shape is None:
            if projection["bias_shape"] is not None:
                raise AttentionContractError(
                    f"fixture.qwen2_attention.attributes.{name}.bias_shape must be null"
                )
        else:
            _require_exact_list(
                projection["bias_shape"],
                f"fixture.qwen2_attention.attributes.{name}.bias_shape",
                bias_shape,
            )
    return attention


def _validate_rotary_evidence(value: object) -> Mapping[str, object]:
    rotary = _mapping_with_exact_keys(
        value,
        "fixture.apply_rotary_pos_emb",
        {"forward_call", "manual_call", "module", "signature"},
    )
    _require_exact_string(
        rotary["module"], "fixture.apply_rotary_pos_emb.module", _EXPECTED_ROPE_MODULE
    )
    _require_exact_string(
        rotary["signature"],
        "fixture.apply_rotary_pos_emb.signature",
        _EXPECTED_ROPE_SIGNATURE,
    )
    for call_name in ("forward_call", "manual_call"):
        call_keys = {"inputs", "return"}
        if call_name == "forward_call":
            call_keys.add("count")
        call = _mapping_with_exact_keys(
            rotary[call_name], f"fixture.apply_rotary_pos_emb.{call_name}", call_keys
        )
        if call_name == "forward_call":
            _require_exact_int(
                call["count"], "fixture.apply_rotary_pos_emb.forward_call.count", 1
            )
        inputs = _mapping_with_exact_keys(
            call["inputs"],
            f"fixture.apply_rotary_pos_emb.{call_name}.inputs",
            {"cos", "k", "q", "sin"},
        )
        _validate_tensor_descriptor(
            inputs["q"],
            f"fixture.apply_rotary_pos_emb.{call_name}.inputs.q",
            [1, 14, 2, 64],
        )
        _validate_tensor_descriptor(
            inputs["k"],
            f"fixture.apply_rotary_pos_emb.{call_name}.inputs.k",
            [1, 2, 2, 64],
        )
        _validate_tensor_descriptor(
            inputs["cos"],
            f"fixture.apply_rotary_pos_emb.{call_name}.inputs.cos",
            [1, 2, 64],
        )
        _validate_tensor_descriptor(
            inputs["sin"],
            f"fixture.apply_rotary_pos_emb.{call_name}.inputs.sin",
            [1, 2, 64],
        )
        returned = _mapping_with_exact_keys(
            call["return"],
            f"fixture.apply_rotary_pos_emb.{call_name}.return",
            {"elements", "kind", "length"},
        )
        _require_exact_string(
            returned["kind"],
            f"fixture.apply_rotary_pos_emb.{call_name}.return.kind",
            "tuple",
        )
        _require_exact_int(
            returned["length"],
            f"fixture.apply_rotary_pos_emb.{call_name}.return.length",
            2,
        )
        if not isinstance(returned["elements"], list) or len(returned["elements"]) != 2:
            raise AttentionContractError(
                f"fixture.apply_rotary_pos_emb.{call_name}.return.elements must contain two tensors"
            )
        _validate_tensor_descriptor(
            returned["elements"][0],
            f"fixture.apply_rotary_pos_emb.{call_name}.return.elements[0]",
            [1, 14, 2, 64],
        )
        _validate_tensor_descriptor(
            returned["elements"][1],
            f"fixture.apply_rotary_pos_emb.{call_name}.return.elements[1]",
            [1, 2, 2, 64],
        )
    return rotary


def _validate_cpu_forward(value: object) -> None:
    evidence = _mapping_with_exact_keys(
        value,
        "fixture.cpu_forward",
        {"batch_size", "input_shape", "return_structure", "sequence_length"},
    )
    _require_exact_int(evidence["batch_size"], "fixture.cpu_forward.batch_size", 1)
    _require_exact_int(
        evidence["sequence_length"], "fixture.cpu_forward.sequence_length", 2
    )
    _require_exact_list(
        evidence["input_shape"], "fixture.cpu_forward.input_shape", [1, 2, HIDDEN_SIZE]
    )
    returned = _mapping_with_exact_keys(
        evidence["return_structure"],
        "fixture.cpu_forward.return_structure",
        {"elements", "kind", "length"},
    )
    _require_exact_string(
        returned["kind"], "fixture.cpu_forward.return_structure.kind", "tuple"
    )
    _require_exact_int(
        returned["length"], "fixture.cpu_forward.return_structure.length", 2
    )
    if not isinstance(returned["elements"], list) or len(returned["elements"]) != 2:
        raise AttentionContractError(
            "fixture.cpu_forward.return_structure.elements must contain two values"
        )
    _validate_tensor_descriptor(
        returned["elements"][0],
        "fixture.cpu_forward.return_structure.elements[0]",
        [1, 2, HIDDEN_SIZE],
    )
    _validate_tensor_descriptor(
        returned["elements"][1],
        "fixture.cpu_forward.return_structure.elements[1]",
        [1, 14, 2, 2],
    )


def _validate_fixture_position_embeddings(value: object) -> None:
    evidence = _mapping_with_exact_keys(
        value, "fixture.position_embeddings", {"arity", "elements"}
    )
    _require_exact_int(evidence["arity"], "fixture.position_embeddings.arity", 2)
    if not isinstance(evidence["elements"], list) or len(evidence["elements"]) != 2:
        raise AttentionContractError(
            "fixture.position_embeddings.elements must contain two tensors"
        )
    for index, descriptor in enumerate(evidence["elements"]):
        _validate_tensor_descriptor(
            descriptor,
            f"fixture.position_embeddings.elements[{index}]",
            [1, 2, HEAD_DIM],
        )


def _validate_tensor_descriptor(
    value: object, name: str, expected_shape: list[int]
) -> None:
    descriptor = _mapping_with_exact_keys(
        value, name, {"device", "dtype", "kind", "shape"}
    )
    _require_exact_string(descriptor["kind"], f"{name}.kind", "tensor")
    _require_exact_string(descriptor["device"], f"{name}.device", "cpu")
    _require_exact_string(descriptor["dtype"], f"{name}.dtype", "torch.float32")
    _require_exact_list(descriptor["shape"], f"{name}.shape", expected_shape)


def _require_exact_list(value: object, name: str, expected: list[int]) -> None:
    if not isinstance(value, list) or value != expected:
        raise AttentionContractError(f"{name} must equal {expected!r}")
    if any((isinstance(item, bool) or not isinstance(item, int) for item in value)):
        raise AttentionContractError(f"{name} must contain integers")


_INTEGER_DTYPES = frozenset(
    {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
)
