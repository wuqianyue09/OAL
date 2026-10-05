"""Private observation-only runtime witnesses for production evidence.

This module deliberately has no dependency on capability admission or reviewed
records.  Its session is meaningful only to the already-authenticated evidence
child: a normal grouped-quadratic dispatch does not create it, read it, or gain
any authorization from it.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType

import torch


def _canonical_json_mapping(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be canonical JSON text")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not JSON") from error
    if not isinstance(decoded, Mapping):
        raise TypeError(f"{name} must encode a JSON object")
    canonical = json.dumps(decoded, sort_keys=True, separators=(",", ":"))
    if value != canonical:
        raise ValueError(f"{name} must use canonical JSON")
    return canonical


@dataclass(frozen=True)
class ProductionWitnessProbe:
    """One deterministic in-bounds coordinate shared by evidence kernels."""

    batch_index: int
    key_value_head: int
    query_head: int
    token_index: int
    pair_index: int
    channel_index: int

    def __post_init__(self) -> None:
        for name, value in (
            ("batch_index", self.batch_index),
            ("key_value_head", self.key_value_head),
            ("query_head", self.query_head),
            ("token_index", self.token_index),
            ("pair_index", self.pair_index),
            ("channel_index", self.channel_index),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"production witness {name} must be a non-negative integer"
                )


@dataclass(frozen=True)
class ProductionWitnessRequest:
    """Immutable base-plan identity and expected capture names for one session."""

    stage_key_json: Mapping[str, str]
    expected_capture_names: tuple[str, ...]
    probe: ProductionWitnessProbe
    stage_probes: Mapping[str, ProductionWitnessProbe] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.stage_key_json, Mapping) or not self.stage_key_json:
            raise ValueError("production witness requires non-empty stage-key JSON")
        canonical_stage_keys: dict[str, str] = {}
        for stage, key_json in self.stage_key_json.items():
            if not isinstance(stage, str) or not stage:
                raise TypeError(
                    "production witness stage names must be non-empty strings"
                )
            canonical_stage_keys[stage] = _canonical_json_mapping(
                key_json, name=f"stage_key_json[{stage!r}]"
            )
        names = tuple(self.expected_capture_names)
        if (
            not names
            or len(set(names)) != len(names)
            or any(not isinstance(name, str) or not name for name in names)
        ):
            raise ValueError(
                "production witness capture names must be unique non-empty strings"
            )
        if not isinstance(self.probe, ProductionWitnessProbe):
            raise TypeError("production witness probe must be a ProductionWitnessProbe")
        if not isinstance(self.stage_probes, Mapping):
            raise TypeError("production witness stage probes must be a mapping")
        canonical_stage_probes: dict[str, ProductionWitnessProbe] = {}
        for stage, probe in self.stage_probes.items():
            if stage not in canonical_stage_keys:
                raise ValueError(
                    "production witness stage probe has no matching stage key"
                )
            if not isinstance(probe, ProductionWitnessProbe):
                raise TypeError(
                    "production witness stage probes must be ProductionWitnessProbe values"
                )
            canonical_stage_probes[stage] = probe
        object.__setattr__(
            self, "stage_key_json", MappingProxyType(canonical_stage_keys)
        )
        object.__setattr__(self, "expected_capture_names", names)
        object.__setattr__(
            self, "stage_probes", MappingProxyType(canonical_stage_probes)
        )

    def probe_for(self, stage: str) -> ProductionWitnessProbe:
        """Return a stage-specific coordinate or the common default coordinate."""
        if not isinstance(stage, str) or stage not in self.stage_key_json:
            raise ValueError("production witness stage has no base plan key")
        return self.stage_probes.get(stage, self.probe)


@dataclass(frozen=True)
class ProductionWitnessCapture:
    """Digest-only description of a concrete device-written witness tensor."""

    name: str
    tensor_sha256: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("production witness capture name must be non-empty")
        if not isinstance(self.tensor_sha256, str) or len(self.tensor_sha256) != 64:
            raise ValueError("production witness capture must have a SHA-256 digest")
        if any(
            not isinstance(dimension, int) or dimension < 0 for dimension in self.shape
        ):
            raise ValueError(
                "production witness capture shape must contain non-negative integers"
            )
        if not isinstance(self.dtype, str) or not self.dtype:
            raise ValueError("production witness capture dtype must be non-empty")


class ProductionWitnessSession:
    """One-shot sink for small, explicitly requested device probe tensors."""

    def __init__(self, request: ProductionWitnessRequest) -> None:
        if not isinstance(request, ProductionWitnessRequest):
            raise TypeError("production witness session requires a request")
        self._request = request
        self._captures: dict[str, ProductionWitnessCapture] = {}
        self._closed = False

    @property
    def request(self) -> ProductionWitnessRequest:
        return self._request

    def expects(self, name: str) -> bool:
        """Return whether this one-shot request explicitly names ``name``."""
        return name in self._request.expected_capture_names

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("production witness session is closed")

    def capture_device_tensor(
        self, name: str, tensor: torch.Tensor
    ) -> ProductionWitnessCapture:
        """Record a digest of a compact, contiguous device-written probe.

        The caller must provide a direct probe buffer; this method never makes
        a contiguous GPU copy.  The host byte transfer is evidence-only and
        occurs after the instrumented CUDA execution has completed.
        """
        self._require_open()
        if name not in self._request.expected_capture_names:
            raise ValueError(f"unexpected production witness capture {name!r}")
        if name in self._captures:
            raise RuntimeError(
                f"production witness capture {name!r} is already captured"
            )
        if not isinstance(tensor, torch.Tensor) or not tensor.is_contiguous():
            raise TypeError("production witness capture must be a contiguous tensor")
        raw_bytes = tensor.detach().view(torch.uint8).cpu().numpy().tobytes()
        capture = ProductionWitnessCapture(
            name=name,
            tensor_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            shape=tuple(int(dimension) for dimension in tensor.shape),
            dtype=str(tensor.dtype).removeprefix("torch."),
        )
        self._captures[name] = capture
        return capture

    def require_complete_captures(self) -> Mapping[str, ProductionWitnessCapture]:
        self._require_open()
        missing = set(self._request.expected_capture_names).difference(self._captures)
        if missing:
            raise RuntimeError(
                "production witness session is missing captures: "
                + ", ".join(sorted(missing))
            )
        return MappingProxyType(dict(self._captures))

    def close(self) -> None:
        self._closed = True


_ACTIVE_PRODUCTION_WITNESS_SESSION: ContextVar[ProductionWitnessSession | None] = (
    ContextVar("grouped_quadratic_production_witness_session", default=None)
)


def active_production_witness_session() -> ProductionWitnessSession | None:
    """Return the current observation sink, never an admission capability."""
    return _ACTIVE_PRODUCTION_WITNESS_SESSION.get()


@contextmanager
def reactivate_production_witness_session(
    session: ProductionWitnessSession,
) -> Iterator[None]:
    """Temporarily expose one existing observer in a custom-autograd callback.

    PyTorch's autograd engine may execute ``Function.backward`` in a context
    that does not inherit the forward call's :class:`ContextVar` values.  This
    bridge reuses the already-open observer instance; it neither creates a
    session nor carries any capability/admission authority.
    """
    if not isinstance(session, ProductionWitnessSession):
        raise TypeError("production witness reactivation requires a session")
    session._require_open()
    active = active_production_witness_session()
    if active is not None and active is not session:
        raise RuntimeError("a different production witness session is already active")
    if active is session:
        yield
        return
    token = _ACTIVE_PRODUCTION_WITNESS_SESSION.set(session)
    try:
        yield
    finally:
        _ACTIVE_PRODUCTION_WITNESS_SESSION.reset(token)


@contextmanager
def activate_production_witness(
    request: ProductionWitnessRequest,
) -> Iterator[ProductionWitnessSession]:
    """Install one non-nestable evidence-only observation session."""
    if active_production_witness_session() is not None:
        raise RuntimeError("production witness sessions cannot be nested")
    session = ProductionWitnessSession(request)
    token = _ACTIVE_PRODUCTION_WITNESS_SESSION.set(session)
    try:
        yield session
    finally:
        session.close()
        _ACTIVE_PRODUCTION_WITNESS_SESSION.reset(token)


__all__ = (
    "ProductionWitnessCapture",
    "ProductionWitnessProbe",
    "ProductionWitnessRequest",
    "ProductionWitnessSession",
    "activate_production_witness",
    "active_production_witness_session",
    "reactivate_production_witness_session",
)
