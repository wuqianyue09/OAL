"""Durable run status and JSONL telemetry for the LoRA pilot."""

from __future__ import annotations
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import stat
import time
from typing import Literal, Protocol, cast, runtime_checkable
from .paths import (
    SCHEMA_VERSION,
    atomic_write_json,
    canonical_json,
    fsync_directory,
    utc_timestamp,
)

RunState = Literal[
    "created", "running", "trained", "evaluating", "completed", "failed", "interrupted"
]
_IN_PROGRESS_STATES = frozenset(("created", "running", "trained", "evaluating"))
_TERMINAL_STATES = frozenset(("completed", "failed", "interrupted"))
_ORDERED_TRANSITIONS: dict[str, frozenset[str]] = {
    "created": frozenset(("running",)),
    "running": frozenset(("trained",)),
    "trained": frozenset(("evaluating",)),
    "evaluating": frozenset(("completed",)),
}
_ALL_STATES = frozenset((*_IN_PROGRESS_STATES, *_TERMINAL_STATES))
_CANONICAL_UTC_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
EVALUATION_SOURCE_DRIFT_FAILURE_MESSAGE = (
    "evaluation source hashes do not match immutable run evidence"
)
EVALUATION_CONFIG_MISMATCH_FAILURE_MESSAGE = (
    "evaluation config does not match effective_config.json"
)
EVALUATION_RUNTIME_SCOPE_FAILURE_MESSAGE = (
    "final evaluation runtime evidence does not match frozen training semantics"
)
COHORT_MEMBER_LOCK_FILENAME = ".stage_d_inventory.lock"
COHORT_FROZEN_FILENAME = ".stage_d_frozen.json"


class RunStateTransitionError(ValueError):
    """Raised when a persisted run status would violate the pilot protocol."""


@runtime_checkable
class RunStateSink(Protocol):
    """The narrow lifecycle surface consumed by training code.

    Concrete stores retain ownership of atomic persistence and state-machine
    validation.  Decorators may add immutable run identity without pretending
    to be a :class:`RunStateStore` subclass.
    """

    def create(
        self, *, run_id: str | None = None, **details: object
    ) -> Mapping[str, object]: ...

    def read(self) -> Mapping[str, object]: ...

    def transition(
        self,
        target_state: RunState,
        *,
        failure: BaseException | Mapping[str, object] | None = None,
        **details: object,
    ) -> Mapping[str, object]: ...


def require_run_state_sink(value: object) -> RunStateSink:
    """Reject a nominal or partial status object before training side effects."""
    required_methods = ("create", "read", "transition")
    if any((not callable(getattr(value, method, None)) for method in required_methods)):
        raise TypeError(
            "status_store must provide callable create, read, and transition methods"
        )
    return cast(RunStateSink, value)


class IdentityRunStateSink:
    """Decorate a state sink with immutable, verified kernel tuning lifecycle identity."""

    def __init__(self, delegate: RunStateSink, identity: Mapping[str, object]) -> None:
        self._delegate = require_run_state_sink(delegate)
        if not isinstance(identity, Mapping) or not identity:
            raise ValueError("fixed identity must be a non-empty mapping")
        self._identity = dict(identity)
        _reject_reserved_status_fields(self._identity)

    def create(
        self, *, run_id: str | None = None, **details: object
    ) -> Mapping[str, object]:
        created = self._delegate.create(run_id=run_id, **self._merge_details(details))
        return self._require_persisted_identity(created)

    def read(self) -> Mapping[str, object]:
        return self._require_persisted_identity(self._delegate.read())

    def transition(
        self,
        target_state: RunState,
        *,
        failure: BaseException | Mapping[str, object] | None = None,
        **details: object,
    ) -> Mapping[str, object]:
        self.read()
        transitioned = self._delegate.transition(
            target_state, failure=failure, **self._merge_details(details)
        )
        return self._require_persisted_identity(transitioned)

    def _merge_details(self, details: Mapping[str, object]) -> dict[str, object]:
        merged = dict(details)
        for key, expected in self._identity.items():
            if key in merged and merged[key] != expected:
                raise ValueError(f"status detail {key!r} conflicts with fixed identity")
            merged[key] = expected
        return merged

    def _require_persisted_identity(
        self, record: Mapping[str, object]
    ) -> Mapping[str, object]:
        for key, expected in self._identity.items():
            if record.get(key) != expected:
                raise ValueError(
                    f"persisted status identity drifted or is missing field {key!r}"
                )
        return record


@contextmanager
def cohort_member_lock(
    run_dir: str | Path,
    *,
    timeout_seconds: float | None = None,
    _allow_frozen_member: bool = False,
):
    """Serialize every mutation of one formal cohort member's source evidence.

    formal cohort inventory creation acquires these same locks in a deterministic
    global order.  A writer therefore either completes before the cohort is
    frozen or observes the freeze marker and fails without changing evidence.
    """
    destination = Path(run_dir).resolve()
    if not destination.is_dir():
        raise FileNotFoundError(
            f"formal cohort member run directory does not exist: {destination}"
        )
    if type(_allow_frozen_member) is not bool:
        raise TypeError("_allow_frozen_member must be a boolean")
    if not _allow_frozen_member:
        require_cohort_member_mutable(destination)
    if timeout_seconds is not None and (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or timeout_seconds < 0
    ):
        raise TypeError("timeout_seconds must be a non-negative number or None")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination / COHORT_MEMBER_LOCK_FILENAME, flags, 384)
    handle = os.fdopen(descriptor, "r+b")
    deadline = (
        None if timeout_seconds is None else time.monotonic() + float(timeout_seconds)
    )
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if deadline is not None and time.monotonic() >= deadline:
                    raise RunStateTransitionError(
                        "timed out acquiring the formal cohort member evidence lock"
                    ) from exc
                time.sleep(0.05)
        if not _allow_frozen_member:
            require_cohort_member_mutable(destination)
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()


def cohort_frozen_marker_path(run_dir: str | Path) -> Path:
    """Return the immutable marker that prevents post-freeze source writes."""
    return Path(run_dir).resolve() / COHORT_FROZEN_FILENAME


def require_cohort_member_mutable(run_dir: str | Path) -> None:
    """Reject a source-evidence mutation after a formal cohort cohort froze it."""
    marker = cohort_frozen_marker_path(run_dir)
    try:
        marker_stat = marker.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(marker_stat.st_mode) or not stat.S_ISREG(marker_stat.st_mode):
        raise RunStateTransitionError(
            "formal cohort freeze marker must be a regular non-symlink file"
        )
    raise RunStateTransitionError(
        "formal cohort member evidence is frozen and cannot be mutated"
    )


class RunStateStore:
    """Atomically persist the one permitted run-state machine in ``status.json``."""

    def __init__(self, status_path: str | Path) -> None:
        self.path = Path(status_path).resolve()

    def create(
        self, *, run_id: str | None = None, **details: object
    ) -> dict[str, object]:
        """Create an initial ``created`` status; existing statuses are never clobbered."""
        _reject_reserved_status_fields(details)
        status: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "updated_at": utc_timestamp(),
            "state": "created",
        }
        if run_id is not None:
            if not isinstance(run_id, str) or not run_id:
                raise ValueError("run_id must be a non-empty string when provided")
            status["run_id"] = run_id
        status.update(details)
        with self._mutation_lock():
            if self.path.exists():
                raise FileExistsError(f"run status already exists: {self.path}")
            atomic_write_json(self.path, status)
        return status

    def read(self) -> dict[str, object]:
        """Load and validate the persisted status before a transition."""
        with self._lock():
            return self._read_unlocked()

    def _read_unlocked(self) -> dict[str, object]:
        """Read and validate while the caller holds this store's lock."""
        if not self.path.is_file():
            raise FileNotFoundError(f"run status file does not exist: {self.path}")
        try:
            with self.path.open(encoding="utf-8") as status_file:
                status = json.load(status_file)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"run status JSON is invalid at {self.path}: {exc.msg}"
            ) from exc
        if not isinstance(status, dict):
            raise ValueError(f"run status must be a JSON object: {self.path}")
        _validate_status_record(status, self.path)
        return status

    def transition(
        self,
        target_state: RunState,
        *,
        failure: BaseException | Mapping[str, object] | None = None,
        **details: object,
    ) -> dict[str, object]:
        """Atomically move to the allowed next state or an in-progress failure."""
        if target_state not in _ALL_STATES:
            raise RunStateTransitionError(f"unknown target run state: {target_state!r}")
        _reject_reserved_status_fields(details)
        with self._mutation_lock():
            status = self._read_unlocked()
            current_state = status["state"]
            assert isinstance(current_state, str)
            if current_state in _TERMINAL_STATES:
                raise RunStateTransitionError(
                    f"{current_state} is terminal and cannot transition to {target_state}"
                )
            allowed = set(_ORDERED_TRANSITIONS[current_state]) | {
                "failed",
                "interrupted",
            }
            if target_state not in allowed:
                raise RunStateTransitionError(
                    f"{current_state} -> {target_state} is not allowed; allowed states: {', '.join(sorted(allowed))}"
                )
            if target_state == "failed":
                if failure is None:
                    raise ValueError(
                        "failure details are required when transitioning to failed"
                    )
                timestamp = utc_timestamp()
                status["failure"] = _failure_record(failure, timestamp)
            elif failure is not None:
                raise ValueError(
                    "failure details are only valid when transitioning to failed"
                )
            else:
                timestamp = utc_timestamp()
            status["schema_version"] = SCHEMA_VERSION
            status["updated_at"] = timestamp
            status["state"] = target_state
            status.update(details)
            atomic_write_json(self.path, status)
            return status

    def reopen_for_resume(self) -> dict[str, object]:
        """Explicitly reopen an interrupted run before strict checkpoint restore.

        This is intentionally separate from :meth:`transition`: terminal
        states remain terminal for ordinary state-machine transitions.  Only a
        recorded ``interrupted`` run may be resumed, and its prior state is
        retained as durable provenance rather than silently overwritten.
        """
        with self._mutation_lock():
            status = self._read_unlocked()
            previous_state = status["state"]
            if previous_state != "interrupted":
                raise RunStateTransitionError(
                    f"resume requires status interrupted; found {previous_state!r}"
                )
            previous_count = status.get("resume_count", 0)
            if type(previous_count) is not int or previous_count < 0:
                raise ValueError(
                    "resume_count must be a non-negative integer when present"
                )
            status.pop("failure", None)
            status["schema_version"] = SCHEMA_VERSION
            status["updated_at"] = utc_timestamp()
            status["state"] = "created"
            status["resumed_from_state"] = previous_state
            status["resume_count"] = previous_count + 1
            atomic_write_json(self.path, status)
            return status

    def reopen_for_evaluation_source_drift(self) -> dict[str, object]:
        """Reopen only the legacy final-evaluation source-hash rejection.

        This is deliberately narrower than a general failed-run recovery.  The
        orchestration layer first proves that no evaluation output exists and
        that the operator explicitly allowed source drift.  The original
        terminal failure remains embedded in append-only recovery history.
        """
        return self._reopen_failed_evaluation(
            expected_message=EVALUATION_SOURCE_DRIFT_FAILURE_MESSAGE,
            reason="allow_source_drift",
            label="source-drift",
        )

    def reopen_for_evaluation_config_mismatch(self) -> dict[str, object]:
        """Recover a trained run poisoned by the preflight-status bug.

        Older evaluators incorrectly turned a completed run terminal when a
        caller supplied a config that did not equal ``effective_config.json``.
        The orchestration layer now proves equality before invoking this narrow
        recovery, so no arbitrary failed training run can be reopened.
        """
        return self._reopen_failed_evaluation(
            expected_message=EVALUATION_CONFIG_MISMATCH_FAILURE_MESSAGE,
            reason="legacy_config_mismatch",
            label="config-mismatch",
        )

    def reopen_for_evaluation_runtime_scope_failure(self) -> dict[str, object]:
        """Recover only an evaluation whose frozen runtime identity now matches."""
        return self._reopen_failed_evaluation(
            expected_message=EVALUATION_RUNTIME_SCOPE_FAILURE_MESSAGE,
            reason="legacy_runtime_scope_mismatch",
            label="runtime-scope",
        )

    def reopen_for_hd_piqa_admission_failure(self) -> dict[str, object]:
        """Caller first verifies completed pilot NLL and its unchanged checkpoint."""
        return self._reopen_failed_evaluation(
            expected_exception_type="RuntimeError",
            expected_message="contraction signature is not in the admitted capability table",
            reason="hd_piqa_capability_inventory",
            label="HD PIQA admission",
        )

    def _reopen_failed_evaluation(
        self,
        *,
        expected_message: str,
        reason: str,
        label: str,
        expected_exception_type: str = "OrchestrationError",
    ) -> dict[str, object]:
        with self._mutation_lock():
            status = self._read_unlocked()
            if status.get("state") != "failed":
                raise RunStateTransitionError(
                    f"evaluation {label} recovery requires status failed"
                )
            failure = status.get("failure")
            if not isinstance(failure, Mapping) or (
                failure.get("exception_type") != expected_exception_type
                or failure.get("exception_message") != expected_message
            ):
                raise RunStateTransitionError(
                    f"evaluation {label} recovery requires the exact eligible evaluation failure"
                )
            history = status.get("evaluation_recovery_history", [])
            if not isinstance(history, list) or any(
                (not isinstance(item, Mapping) for item in history)
            ):
                raise ValueError(
                    "evaluation_recovery_history must be a list of objects"
                )
            previous_count = status.get("evaluation_recovery_count", 0)
            if type(previous_count) is not int or previous_count != len(history):
                raise ValueError(
                    "evaluation_recovery_count must match evaluation_recovery_history"
                )
            timestamp = utc_timestamp()
            history = [
                *(dict(item) for item in history),
                {"failure": dict(failure), "recovered_at": timestamp, "reason": reason},
            ]
            status.pop("failure", None)
            status["schema_version"] = SCHEMA_VERSION
            status["updated_at"] = timestamp
            status["state"] = "trained"
            status["evaluation_recovery_count"] = previous_count + 1
            status["evaluation_recovery_history"] = history
            atomic_write_json(self.path, status)
            return status

    @contextmanager
    def _lock(self):
        """Take the POSIX interprocess lock associated with this status path."""
        lock_path = self.path.with_name(f".{self.path.name}.lock")
        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def _mutation_lock(self):
        """Hold the member lock before changing a source-evidence status file."""
        with cohort_member_lock(self.path.parent):
            require_cohort_member_mutable(self.path.parent)
            with self._lock():
                yield


def create_run_status(
    status_path: str | Path, *, run_id: str | None = None, **details: object
) -> dict[str, object]:
    """Create a status file without exposing the store implementation."""
    return RunStateStore(status_path).create(run_id=run_id, **details)


def transition_run_status(
    status_path: str | Path,
    target_state: RunState,
    *,
    failure: BaseException | Mapping[str, object] | None = None,
    **details: object,
) -> dict[str, object]:
    """Transition a persisted status file using :class:`RunStateStore`."""
    return RunStateStore(status_path).transition(
        target_state, failure=failure, **details
    )


class JsonlWriter:
    """Append one fsynced, canonical telemetry event per JSONL line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, record: Mapping[str, object]) -> dict[str, object]:
        """Append ``record`` with the required schema version and UTC timestamp."""
        if not isinstance(record, Mapping):
            raise TypeError("JSONL telemetry record must be a mapping")
        if "schema_version" in record and record["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"JSONL schema_version must be {SCHEMA_VERSION}")
        if "timestamp" in record:
            raise ValueError("JSONL timestamp is assigned by JsonlWriter")
        if not self.path.parent.exists():
            raise FileNotFoundError(
                f"JSONL target directory does not exist: {self.path.parent}"
            )
        if not self.path.parent.is_dir():
            raise NotADirectoryError(
                f"JSONL target parent is not a directory: {self.path.parent}"
            )
        event = dict(record)
        event["schema_version"] = SCHEMA_VERSION
        event["timestamp"] = utc_timestamp()
        encoded = (canonical_json(event) + "\n").encode("utf-8")
        created = not self.path.exists()
        with self.path.open("ab") as metrics_file:
            metrics_file.write(encoded)
            metrics_file.flush()
            os.fsync(metrics_file.fileno())
        if created:
            try:
                fsync_directory(self.path.parent)
            except OSError as exc:
                raise OSError(
                    f"JSONL record was written and file-fsynced, but target directory could not be fsynced: {self.path.parent}"
                ) from exc
        return event

    append = write


def write_run_manifest(
    path: str | Path, record: Mapping[str, object]
) -> dict[str, object]:
    """Write a schema-versioned, timestamped non-JSONL manifest atomically."""
    if not isinstance(record, Mapping):
        raise TypeError("run manifest must be a mapping")
    manifest = dict(record)
    if "schema_version" in manifest and manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"run manifest schema_version must be {SCHEMA_VERSION}")
    manifest["schema_version"] = SCHEMA_VERSION
    if "recorded_at" in manifest:
        _validate_utc_timestamp(manifest["recorded_at"], "recorded_at", canonical=True)
    else:
        manifest["recorded_at"] = utc_timestamp()
    atomic_write_json(path, manifest)
    return manifest


def _failure_record(
    failure: BaseException | Mapping[str, object], timestamp: str
) -> dict[str, object]:
    if isinstance(failure, BaseException):
        return {
            "exception_type": type(failure).__name__,
            "exception_message": str(failure),
            "timestamp": timestamp,
        }
    exception_type = failure.get("exception_type")
    exception_message = failure.get("exception_message")
    if not isinstance(exception_type, str) or not exception_type:
        raise ValueError("failure details require a non-empty exception_type")
    if not isinstance(exception_message, str):
        raise ValueError("failure details require a string exception_message")
    result = dict(failure)
    result["timestamp"] = timestamp
    return result


def _reject_reserved_status_fields(details: Mapping[str, object]) -> None:
    reserved = sorted(
        {"schema_version", "state", "updated_at", "failure"} & set(details)
    )
    if reserved:
        raise ValueError(
            "status detail fields may not override reserved field(s): "
            + ", ".join(reserved)
        )


def _validate_status_record(status: Mapping[str, object], status_path: Path) -> None:
    schema_version = status.get("schema_version")
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"run status schema_version must be exactly integer {SCHEMA_VERSION}: {status_path}"
        )
    _validate_utc_timestamp(status.get("updated_at"), "updated_at")
    state = status.get("state")
    if state not in _ALL_STATES:
        raise ValueError(f"run status has unknown state {state!r}: {status_path}")
    if state == "failed":
        _validate_failure_record(status.get("failure"), status["updated_at"])
    elif "failure" in status:
        raise ValueError(
            "run status failure details are valid only when state is failed"
        )


def _validate_failure_record(failure: object, updated_at: object) -> None:
    if not isinstance(failure, Mapping):
        raise ValueError("failed run status requires a failure object")
    exception_type = failure.get("exception_type")
    if not isinstance(exception_type, str) or not exception_type:
        raise ValueError("failure.exception_type must be a non-empty string")
    exception_message = failure.get("exception_message")
    if not isinstance(exception_message, str):
        raise ValueError("failure.exception_message must be a string")
    timestamp = failure.get("timestamp")
    _validate_utc_timestamp(timestamp, "failure.timestamp")
    if timestamp != updated_at:
        raise ValueError("failure.timestamp must equal updated_at")


def _validate_utc_timestamp(
    value: object, field_name: str, *, canonical: bool = False
) -> None:
    if not isinstance(value, str) or not value.endswith("Z") or "T" not in value:
        raise ValueError(f"{field_name} must be a UTC ISO-8601 timestamp ending in 'Z'")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be a UTC ISO-8601 timestamp ending in 'Z'"
        ) from exc
    if canonical and parsed.strftime(_CANONICAL_UTC_FORMAT) != value:
        raise ValueError(
            f"{field_name} must use canonical UTC format {_CANONICAL_UTC_FORMAT!r}"
        )
