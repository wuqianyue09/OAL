"""Process-local, observation-only traces for grouped quadratic execution."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json

from torch.profiler import record_function

_PHYSICAL_STAGE_RANGE_NAMES = {
    "forward": "hd_mgq.public.forward",
    "normalization": "hd_mgq.stage.normalization",
    "prefix": "hd_mgq.stage.prefix",
    "suffix": "hd_mgq.stage.suffix",
}
_BACKWARD_RANGE_NAME = "hd_mgq.public.backward"


@dataclass(frozen=True)
class GroupedExecutionPhysicalStage:
    """One observed physical grouped launch and its exact canonical plan payload."""

    call_ordinal: int
    stage: str
    plan: str
    plan_id: str
    physical_path: str
    requested_gradient_mask: tuple[bool, bool, bool, bool, bool, bool]
    plan_payload_sha256: str


@dataclass(frozen=True)
class GroupedExecutionBackwardEnvelope:
    """The public backward scope linked to its ordered physical child launches."""

    call_ordinal: int
    range_identity: str
    child_physical_stage_ordinals: tuple[int, ...]
    ordered_stage_plan_bundle_sha256: str


@dataclass(frozen=True)
class GroupedExecutionTrace:
    """Immutable result of one completed grouped execution observation session."""

    physical_stages: tuple[GroupedExecutionPhysicalStage, ...]
    backward_envelopes: tuple[GroupedExecutionBackwardEnvelope, ...]


def _default_range_factory(name: str):
    return record_function(name)


def _canonical_plan_payload(plan_payload: object) -> tuple[
    str,
    str,
    str,
    tuple[bool, bool, bool, bool, bool, bool],
]:
    """Validate and compactly encode the exact plan object already selected to launch."""
    if not isinstance(plan_payload, dict):
        raise TypeError("grouped execution plan payload must be a JSON object")
    try:
        plan = json.dumps(
            plan_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        decoded = json.loads(plan)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise TypeError(
            "grouped execution plan payload must be JSON serializable"
        ) from error
    if not isinstance(decoded, dict):
        raise TypeError("grouped execution plan payload must be a JSON object")
    plan_id = decoded.get("plan_id")
    physical_path = decoded.get("physical_path")
    geometry = decoded.get("geometry")
    if not isinstance(plan_id, str) or not plan_id:
        raise ValueError("grouped execution plan payload must contain plan_id")
    if not isinstance(physical_path, str) or not physical_path:
        raise ValueError("grouped execution plan payload must contain physical_path")
    if not isinstance(geometry, dict):
        raise ValueError("grouped execution plan payload must contain geometry")
    requested_gradient_mask = geometry.get("requested_gradient_mask")
    if (
        not isinstance(requested_gradient_mask, list)
        or len(requested_gradient_mask) != 6
        or not all(isinstance(value, bool) for value in requested_gradient_mask)
    ):
        raise ValueError(
            "grouped execution plan payload must contain six requested gradient booleans"
        )
    return plan, plan_id, physical_path, tuple(requested_gradient_mask)  # type: ignore[return-value]


def _canonical_stage_plan_bundle(
    physical_stages: tuple[GroupedExecutionPhysicalStage, ...],
) -> str:
    return json.dumps(
        [
            {"stage": physical_stage.stage, "plan": physical_stage.plan}
            for physical_stage in physical_stages
        ],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


class _GroupedExecutionTraceSession:
    """Mutable sink whose only output is the immutable trace value above."""

    def __init__(self, range_factory: object) -> None:
        if not callable(range_factory):
            raise TypeError("range_factory must be callable")
        self._range_factory = range_factory
        self._physical_stages: list[GroupedExecutionPhysicalStage] = []
        self._backward_envelopes: list[GroupedExecutionBackwardEnvelope] = []
        self._next_call_ordinal = 1
        self._closed = False
        self._finished_trace: GroupedExecutionTrace | None = None

    @property
    def is_open(self) -> bool:
        return not self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("grouped execution trace session is closed")

    def _next_ordinal(self) -> int:
        ordinal = self._next_call_ordinal
        self._next_call_ordinal += 1
        return ordinal

    @contextmanager
    def observe_physical_stage(self, stage: str, plan_payload_factory: object):
        self._require_open()
        range_name = _PHYSICAL_STAGE_RANGE_NAMES.get(stage)
        if range_name is None:
            raise ValueError(
                "grouped execution stage must be forward, normalization, prefix, or suffix"
            )
        if not callable(plan_payload_factory):
            raise TypeError("grouped execution plan payload factory must be callable")
        plan_payload = plan_payload_factory()
        plan, plan_id, physical_path, requested_gradient_mask = _canonical_plan_payload(
            plan_payload
        )
        record = GroupedExecutionPhysicalStage(
            call_ordinal=self._next_ordinal(),
            stage=stage,
            plan=plan,
            plan_id=plan_id,
            physical_path=physical_path,
            requested_gradient_mask=requested_gradient_mask,
            plan_payload_sha256=hashlib.sha256(plan.encode("utf-8")).hexdigest(),
        )
        with self._range_factory(range_name):
            yield
        self._physical_stages.append(record)

    @contextmanager
    def observe_backward_envelope(self):
        self._require_open()
        call_ordinal = self._next_ordinal()
        first_child_index = len(self._physical_stages)
        with self._range_factory(_BACKWARD_RANGE_NAME):
            yield
        children = tuple(self._physical_stages[first_child_index:])
        ordered_bundle = _canonical_stage_plan_bundle(children)
        self._backward_envelopes.append(
            GroupedExecutionBackwardEnvelope(
                call_ordinal=call_ordinal,
                range_identity=_BACKWARD_RANGE_NAME,
                child_physical_stage_ordinals=tuple(
                    child.call_ordinal for child in children
                ),
                ordered_stage_plan_bundle_sha256=hashlib.sha256(
                    ordered_bundle.encode("utf-8")
                ).hexdigest(),
            )
        )

    def close(self) -> None:
        self._closed = True

    def finish(self) -> GroupedExecutionTrace:
        if self.is_open:
            raise RuntimeError("grouped execution trace session is still open")
        if self._finished_trace is None:
            raise RuntimeError("grouped execution trace session has not finished")
        return self._finished_trace


_ACTIVE_GROUPED_EXECUTION_TRACE: ContextVar[_GroupedExecutionTraceSession | None] = (
    ContextVar("grouped_execution_trace_session", default=None)
)


def _active_grouped_execution_trace_session() -> _GroupedExecutionTraceSession | None:
    return _ACTIVE_GROUPED_EXECUTION_TRACE.get()


@contextmanager
def _observe_grouped_stage(stage: str, plan_payload_factory: object):
    """Add one profiler scope and metadata record only for an active trace."""
    session = _active_grouped_execution_trace_session()
    if session is None:
        yield
        return
    with session.observe_physical_stage(stage, plan_payload_factory):
        yield


@contextmanager
def _observe_grouped_backward():
    """Add the public backward envelope only for an active trace."""
    session = _active_grouped_execution_trace_session()
    if session is None:
        yield
        return
    with session.observe_backward_envelope():
        yield


@contextmanager
def capture_grouped_execution_trace(*, range_factory: object = _default_range_factory):
    """Capture one non-nestable, process-local observation-only trace."""
    if _active_grouped_execution_trace_session() is not None:
        raise RuntimeError("grouped execution trace sessions cannot be nested")
    session = _GroupedExecutionTraceSession(range_factory)
    token = _ACTIVE_GROUPED_EXECUTION_TRACE.set(session)
    try:
        yield session
    finally:
        session.close()
        session._finished_trace = GroupedExecutionTrace(
            physical_stages=tuple(session._physical_stages),
            backward_envelopes=tuple(session._backward_envelopes),
        )
        _ACTIVE_GROUPED_EXECUTION_TRACE.reset(token)


@contextmanager
def reactivate_grouped_execution_trace(session: object):
    """Temporarily reattach one existing trace across an autograd callback."""
    if not isinstance(session, _GroupedExecutionTraceSession):
        raise TypeError(
            "grouped execution trace reactivation requires an existing trace session"
        )
    session._require_open()
    active = _active_grouped_execution_trace_session()
    if active is not None and active is not session:
        raise RuntimeError(
            "a different grouped execution trace session is already active"
        )
    if active is session:
        yield
        return
    token = _ACTIVE_GROUPED_EXECUTION_TRACE.set(session)
    try:
        yield
    finally:
        _ACTIVE_GROUPED_EXECUTION_TRACE.reset(token)


__all__ = (
    "GroupedExecutionBackwardEnvelope",
    "GroupedExecutionPhysicalStage",
    "GroupedExecutionTrace",
    "capture_grouped_execution_trace",
    "reactivate_grouped_execution_trace",
)
