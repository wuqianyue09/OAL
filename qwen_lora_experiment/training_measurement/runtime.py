"""Optional, observation-only runtime scopes for training measurement.

Normal pilot training receives the module-level no-op runtime.  The active
runtime keeps profiler, CUDA timing, and public operator trace state on this side
of the boundary; callers provide only scalar step numbers and range names.
Profiler range strings keep their historical ``plan_b`` namespace.
"""

from __future__ import annotations
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
import time
from typing import Protocol
import torch

ScopeFactory = Callable[[str], AbstractContextManager[object]]
TraceCapture = Callable[..., AbstractContextManager[object]]
TraceReactivation = Callable[[object], AbstractContextManager[object]]
Clock = Callable[[], float]


class TrainingMeasurementRuntime(Protocol):
    """The narrow, tensor-blind measurement seam used by ``training.py``."""

    def prepare_step(self, step: int) -> None: ...

    def step(self, step: int) -> AbstractContextManager[None]: ...

    def scope(self, name: str) -> AbstractContextManager[None]: ...

    def external_scope(self, name: str) -> AbstractContextManager[None]: ...

    def finish(self) -> object | None: ...


class CheckpointMeasurementContext(Protocol):
    """Only the checkpoint ``context_fn`` needed during model construction."""

    def checkpoint_context_fn(
        self,
    ) -> tuple[AbstractContextManager[None], AbstractContextManager[None]]: ...


class MeasurementRuntimeError(RuntimeError):
    """Raised when an active training measurement runtime cannot preserve trace ownership."""


class _NoOpTrainingMeasurementRuntime:
    """Stable singleton used by default so the hot path has no measurement branches."""

    def prepare_step(self, step: int) -> None:
        del step

    def step(self, step: int) -> AbstractContextManager[None]:
        del step
        return nullcontext()

    def scope(self, name: str) -> AbstractContextManager[None]:
        del name
        return nullcontext()

    def external_scope(self, name: str) -> AbstractContextManager[None]:
        del name
        return nullcontext()

    def finish(self) -> None:
        return None


NOOP_TRAINING_MEASUREMENT_RUNTIME: TrainingMeasurementRuntime = (
    _NoOpTrainingMeasurementRuntime()
)


@dataclass(frozen=True)
class CompletedProfilerWindow:
    """One independently closed profiler window with its complete ownership facts."""

    window_id: str
    sample_step: int | None
    external_scope: str | None
    wall_ms: float
    tokens: int
    allocated_peak_bytes: int
    reserved_peak_bytes: int
    trace: object | None
    profiler_events: tuple[object, ...]


@dataclass(frozen=True)
class ProfilerCollection:
    """Immutable bounded windows; parsing and serialization belong to evidence assembly."""

    windows: tuple[CompletedProfilerWindow, ...]


class ProfilerRuntime:
    """Own optional profiler/CUDA/trace bookkeeping for selected training steps.

    ``warmup_steps`` selects the leading one-based training steps and
    ``sample_steps`` supplies the remaining one-based sample positions.  The
    training loop never receives the profiler, CUDA primitives, or trace
    session, so the seam cannot inspect or mutate model computation.
    """

    def __init__(
        self,
        *,
        device: torch.device | str,
        warmup_steps: int,
        sample_steps: Sequence[int],
        tokens_per_step: int = 1,
        synchronize: Callable[[], None] | None = None,
        clock: Clock = time.perf_counter,
        scope_factory: ScopeFactory | None = None,
        capture_grouped_execution_trace: TraceCapture | None = None,
        reactivate_grouped_execution_trace: TraceReactivation | None = None,
        profiler_factory: Callable[[], AbstractContextManager[object]] | None = None,
    ) -> None:
        if type(warmup_steps) is not int or warmup_steps < 0:
            raise ValueError("warmup_steps must be a non-negative integer")
        normalized_samples = tuple(sample_steps)
        if any((type(step) is not int or step <= 0 for step in normalized_samples)):
            raise ValueError("sample_steps must contain positive integer step numbers")
        if len(set(normalized_samples)) != len(normalized_samples):
            raise ValueError("sample_steps must not contain duplicates")
        if type(tokens_per_step) is not int or tokens_per_step <= 0:
            raise ValueError("tokens_per_step must be a positive integer")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if synchronize is not None and (not callable(synchronize)):
            raise TypeError("synchronize must be callable when supplied")
        if scope_factory is not None and (not callable(scope_factory)):
            raise TypeError("scope_factory must be callable when supplied")
        if profiler_factory is not None and (not callable(profiler_factory)):
            raise TypeError("profiler_factory must be callable when supplied")
        self._device = torch.device(device)
        if any((step <= warmup_steps for step in normalized_samples)):
            raise ValueError("sample_steps must be strictly post-warmup")
        self._warmup_steps = warmup_steps
        self._configured_steps = frozenset(normalized_samples)
        self._tokens_per_step = tokens_per_step
        self._synchronize = synchronize or self._cuda_synchronize
        self._clock = clock
        self._scope_factory = scope_factory or torch.autograd.profiler.record_function
        self._capture_grouped_execution_trace = capture_grouped_execution_trace
        self._reactivate_grouped_execution_trace = reactivate_grouped_execution_trace
        self._profiler_factory = profiler_factory
        self._profiler: AbstractContextManager[object] | None = None
        self._profiler_open = False
        self._prepared_step: int | None = None
        self._last_prepared_step_is_measured = False
        self._prepared_step_has_completed = False
        self._scope_enabled: ContextVar[bool] = ContextVar(
            "measurement_scope_enabled", default=False
        )
        self._open_trace_session: ContextVar[object | None] = ContextVar(
            "measurement_trace_session", default=None
        )
        self._completed_windows: list[CompletedProfilerWindow] = []
        self._next_window_ordinal = 1
        self._finished_collection: ProfilerCollection | None = None

    def prepare_step(self, step: int) -> None:
        self._require_open()
        self._require_step_number(step)
        self._prepared_step = step
        self._last_prepared_step_is_measured = step in self._configured_steps
        self._prepared_step_has_completed = False
        if self._last_prepared_step_is_measured:
            try:
                self._ensure_profiler_open()
                if self._device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self._device)
            except BaseException:
                self._close_profiler()
                raise

    @contextmanager
    def step(self, step: int):
        self._require_open()
        self._require_step_number(step)
        if self._prepared_step != step:
            raise MeasurementRuntimeError(
                "training measurement step scope requires prepare_step for the same step"
            )
        measured = self._last_prepared_step_is_measured
        if not measured:
            try:
                yield
            finally:
                self._prepared_step_has_completed = True
            return
        scope_token = None
        session: object | None = None
        try:
            self._synchronize()
            started_at = self._clock_value()
            scope_token = self._scope_enabled.set(True)
            with self._scope_context(f"plan_b.step.{step}", enabled=True):
                with self._trace_capture()(
                    range_factory=self._scope_factory
                ) as session:
                    trace_token = self._open_trace_session.set(session)
                    try:
                        yield
                    finally:
                        self._open_trace_session.reset(trace_token)
            self._synchronize()
            finished_trace = self._finish_trace(session)
            self._completed_windows.append(
                CompletedProfilerWindow(
                    window_id=self._next_window_id(f"step-{step}"),
                    sample_step=step,
                    external_scope=None,
                    wall_ms=(self._clock_value() - started_at) * 1000.0,
                    tokens=self._tokens_per_step,
                    allocated_peak_bytes=self._max_memory_allocated_bytes(),
                    reserved_peak_bytes=self._max_memory_reserved_bytes(),
                    trace=finished_trace,
                    profiler_events=self._close_profiler(),
                )
            )
        finally:
            if scope_token is not None:
                self._scope_enabled.reset(scope_token)
            self._prepared_step_has_completed = True
            self._close_profiler()

    def scope(self, name: str) -> AbstractContextManager[None]:
        return self._scope_context(name, enabled=self._scope_enabled.get())

    def external_scope(self, name: str) -> AbstractContextManager[None]:
        if not self._last_prepared_step_is_measured:
            return self._scope_context(name, enabled=False)
        if self._scope_enabled.get():
            raise MeasurementRuntimeError(
                "training measurement external scopes must be step-external"
            )
        if not self._prepared_step_has_completed:
            raise MeasurementRuntimeError(
                "training measurement external scopes require the prepared measured step to finish first"
            )
        return self._isolated_external_scope(name)

    @contextmanager
    def _isolated_external_scope(self, name: str):
        """Profile one measured step's external envelope without crossing a gap."""
        scope = self._external_scope_name(name)
        started_at: float | None = None
        try:
            self._ensure_profiler_open()
            self._synchronize()
            started_at = self._clock_value()
            with self._scope_context(name, enabled=True):
                yield
        finally:
            try:
                if started_at is not None:
                    self._synchronize()
                    self._completed_windows.append(
                        CompletedProfilerWindow(
                            window_id=self._next_window_id(f"external-{scope}"),
                            sample_step=None,
                            external_scope=scope,
                            wall_ms=(self._clock_value() - started_at) * 1000.0,
                            tokens=0,
                            allocated_peak_bytes=0,
                            reserved_peak_bytes=0,
                            trace=None,
                            profiler_events=self._close_profiler(),
                        )
                    )
            finally:
                self._close_profiler()

    def checkpoint_context_fn(
        self,
    ) -> tuple[AbstractContextManager[None], AbstractContextManager[None]]:
        """Return original/recompute contexts tied to the current open trace session."""
        self._require_open()
        session = self._open_trace_session.get()
        if session is None:
            return (nullcontext(), nullcontext())
        return (
            self._scope_context("plan_b.checkpoint.original", enabled=True),
            self._checkpoint_recompute_context(session),
        )

    @contextmanager
    def _checkpoint_recompute_context(self, session: object):
        with self._trace_reactivation()(session):
            with self._scope_context("plan_b.checkpoint.recompute", enabled=True):
                yield

    def finish(self) -> ProfilerCollection:
        """Close an optional profiler and expose raw observations exactly once."""
        if self._finished_collection is not None:
            return self._finished_collection
        self._close_profiler()
        self._finished_collection = ProfilerCollection(
            windows=tuple(self._completed_windows)
        )
        return self._finished_collection

    def _scope_context(
        self, name: str, *, enabled: bool
    ) -> AbstractContextManager[None]:
        if type(name) is not str or not name.startswith("plan_b."):
            raise ValueError(
                "training measurement scope names must be strings prefixed with 'plan_b.'"
            )
        if not enabled:
            return nullcontext()
        return self._scope_factory(name)

    @staticmethod
    def _external_scope_name(name: str) -> str:
        if type(name) is not str or not name.startswith("plan_b.external."):
            raise ValueError(
                "training measurement external scopes must use plan_b.external.<scope>"
            )
        scope = name.removeprefix("plan_b.external.")
        if scope not in {"validation", "save", "kernel_diagnostic"}:
            raise ValueError("training measurement external scope is not declared")
        return scope

    def _next_window_id(self, suffix: str) -> str:
        window_id = f"window-{self._next_window_ordinal}-{suffix}"
        self._next_window_ordinal += 1
        return window_id

    def _trace_capture(self) -> TraceCapture:
        if self._capture_grouped_execution_trace is None:
            self._load_public_operator_trace_api()
        assert self._capture_grouped_execution_trace is not None
        return self._capture_grouped_execution_trace

    def _trace_reactivation(self) -> TraceReactivation:
        if self._reactivate_grouped_execution_trace is None:
            self._load_public_operator_trace_api()
        assert self._reactivate_grouped_execution_trace is not None
        return self._reactivate_grouped_execution_trace

    def _load_public_operator_trace_api(self) -> None:
        """Import only the operator package's public trace bridge when active."""
        try:
            from oal_attention import (
                capture_grouped_execution_trace,
                reactivate_grouped_execution_trace,
            )
        except Exception as exc:
            raise MeasurementRuntimeError(
                "training measurement requires the public oal_attention trace API"
            ) from exc
        self._capture_grouped_execution_trace = capture_grouped_execution_trace
        self._reactivate_grouped_execution_trace = reactivate_grouped_execution_trace

    def _ensure_profiler_open(self) -> None:
        if self._profiler is not None or self._profiler_factory is None:
            return
        profiler = self._profiler_factory()
        self._profiler = profiler
        profiler.__enter__()
        self._profiler_open = True

    def _close_profiler(self) -> tuple[object, ...]:
        """Finish one contiguous measured window before any inactive gap."""
        profiler = self._profiler
        if profiler is None:
            return ()
        profiler_events: tuple[object, ...] = ()
        try:
            if self._profiler_open:
                profiler.__exit__(None, None, None)
                self._profiler_open = False
            events = getattr(profiler, "events", None)
            if callable(events):
                raw_events = events()
                if not isinstance(raw_events, Sequence) or isinstance(
                    raw_events, (str, bytes, bytearray)
                ):
                    raise MeasurementRuntimeError(
                        "training measurement profiler events must be a sequence"
                    )
                profiler_events = tuple(raw_events)
        finally:
            self._profiler = None
        return profiler_events

    def _cuda_synchronize(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def _max_memory_allocated_bytes(self) -> int:
        if self._device.type != "cuda":
            return 0
        return int(torch.cuda.max_memory_allocated(self._device))

    def _max_memory_reserved_bytes(self) -> int:
        if self._device.type != "cuda":
            return 0
        return int(torch.cuda.max_memory_reserved(self._device))

    def _clock_value(self) -> float:
        value = self._clock()
        if not isinstance(value, (int, float)):
            raise MeasurementRuntimeError(
                "training measurement clock must return a numeric value"
            )
        return float(value)

    def _require_open(self) -> None:
        if self._finished_collection is not None:
            raise MeasurementRuntimeError(
                "training measurement runtime is already finished"
            )

    @staticmethod
    def _require_step_number(step: int) -> None:
        if type(step) is not int or step <= 0:
            raise ValueError(
                "training measurement step numbers must be positive integers"
            )

    @staticmethod
    def _finish_trace(session: object) -> object:
        finish = getattr(session, "finish", None)
        return finish() if callable(finish) else session


__all__ = (
    "CheckpointMeasurementContext",
    "CompletedProfilerWindow",
    "NOOP_TRAINING_MEASUREMENT_RUNTIME",
    "ProfilerCollection",
    "ProfilerRuntime",
    "MeasurementRuntimeError",
    "TrainingMeasurementRuntime",
)
