"""Canonical create-only sidecars for optional benchmark evaluations."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


@dataclass(frozen=True)
class PublishedSidecar:
    final_path: Path
    predictions_path: Path
    identity_path: Path
    reused: bool


@dataclass(frozen=True)
class ReusableSidecar:
    final_path: Path
    predictions_path: Path
    identity_path: Path
    summary: dict[str, object]
    predictions: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class _ArtifactNames:
    directory: str
    summary: str
    predictions: str
    identity: str


_TASK_NAMES = {
    "arc_easy": _ArtifactNames(
        directory="arc_easy_sidecar_v1",
        summary="pilot_arc_easy_eval.json",
        predictions="arc_easy_test_predictions.jsonl",
        identity="arc_easy_test_predictions_identity.json",
    ),
    "hellaswag": _ArtifactNames(
        directory="hellaswag_sidecar_v2",
        summary="pilot_hellaswag_eval.json",
        predictions="hellaswag_validation_predictions.jsonl",
        identity="hellaswag_validation_predictions_identity.json",
    ),
    "gsm8k": _ArtifactNames(
        directory="gsm8k_sidecar_v2",
        summary="pilot_gsm8k_eval.json",
        predictions="gsm8k_test_predictions.jsonl",
        identity="gsm8k_test_predictions_identity.json",
    ),
}
_PUBLISHED_CONTENT_FIELD = "published_content"


def publish_sidecar(
    run_dir: str | Path,
    *,
    task: str,
    summary: Mapping[str, object],
    predictions: Sequence[Mapping[str, object]],
    identity: Mapping[str, object],
) -> PublishedSidecar:
    """Publish predictions and identity first, then summary as completion marker."""
    names = _TASK_NAMES.get(task)
    if names is None:
        raise ValueError(
            f"supported task must be one of {', '.join(sorted(_TASK_NAMES))}"
        )
    destination = Path(run_dir)
    if not destination.is_dir():
        raise FileNotFoundError(
            f"evaluation run directory does not exist: {destination}"
        )
    sidecar = destination / names.directory
    sidecar.mkdir(mode=448, exist_ok=True)
    if not sidecar.is_dir():
        raise ValueError(f"evaluation sidecar path is not a directory: {sidecar}")
    if _PUBLISHED_CONTENT_FIELD in identity:
        raise ValueError(
            f"sidecar identity field {_PUBLISHED_CONTENT_FIELD!r} is reserved"
        )
    summary_bytes = _canonical_json_bytes(summary) + b"\n"
    prediction_bytes = b"".join(
        (_canonical_json_bytes(row) + b"\n" for row in predictions)
    )
    published_identity = {
        **identity,
        _PUBLISHED_CONTENT_FIELD: {
            "predictions_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
            "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        },
    }
    identity_bytes = _canonical_json_bytes(published_identity) + b"\n"
    predictions_path = sidecar / names.predictions
    identity_path = sidecar / names.identity
    final_path = sidecar / names.summary
    reused = (
        _publish_create_only(predictions_path, prediction_bytes),
        _publish_create_only(identity_path, identity_bytes),
        _publish_create_only(final_path, summary_bytes),
    )
    return PublishedSidecar(
        final_path=final_path,
        predictions_path=predictions_path,
        identity_path=identity_path,
        reused=all(reused),
    )


def load_reusable_sidecar(
    run_dir: str | Path, *, task: str, identity: Mapping[str, object]
) -> ReusableSidecar | None:
    """Fail on an identity conflict, or load one already complete sidecar."""
    names = _task_names(task)
    sidecar, final_path, predictions_path, identity_path = _artifact_paths(
        Path(run_dir), names
    )
    if not sidecar.exists():
        return None
    if not sidecar.is_dir():
        raise ValueError(f"evaluation sidecar path is not a directory: {sidecar}")
    if _PUBLISHED_CONTENT_FIELD in identity:
        raise ValueError(
            f"sidecar identity field {_PUBLISHED_CONTENT_FIELD!r} is reserved"
        )
    expected_identity = _canonical_json_bytes(identity)
    identity_exists = identity_path.is_file()
    predictions_exist = predictions_path.is_file()
    final_exists = final_path.is_file()
    if predictions_exist and (not identity_exists):
        raise ValueError(f"existing {task} sidecar has predictions without identity")
    if final_exists and (not identity_exists):
        raise ValueError(f"completed {task} sidecar is missing its prediction identity")
    if identity_path.exists():
        stored_identity = _read_json_object(
            identity_path, label=f"{task} sidecar prediction identity"
        )
        published_content = stored_identity.pop(_PUBLISHED_CONTENT_FIELD, None)
        if _canonical_json_bytes(stored_identity) != expected_identity:
            raise ValueError(
                f"existing {task} sidecar identity conflicts with this request"
            )
        content_digests = _content_digests(
            published_content, task=task, identity_path=identity_path
        )
        if not predictions_exist:
            raise ValueError(f"existing {task} sidecar identity has no predictions")
        _require_digest(
            predictions_path,
            content_digests["predictions_sha256"],
            label=f"{task} sidecar predictions",
        )
    else:
        content_digests = None
    if not final_exists:
        if identity_exists or predictions_exist:
            raise ValueError(
                f"existing {task} sidecar is incomplete and missing its summary"
            )
        return None
    if not predictions_exist or not identity_exists or content_digests is None:
        raise ValueError(f"completed {task} sidecar is incomplete")
    _require_digest(
        final_path, content_digests["summary_sha256"], label=f"{task} sidecar summary"
    )
    summary = _read_json_object(final_path, label=f"{task} sidecar summary")
    predictions = _read_jsonl(predictions_path, label=f"{task} sidecar predictions")
    return ReusableSidecar(
        final_path=final_path,
        predictions_path=predictions_path,
        identity_path=identity_path,
        summary=summary,
        predictions=tuple(predictions),
    )


@contextmanager
def sidecar_evaluation_session(run_dir: str | Path, *, task: str):
    """Prevent two processes from evaluating and publishing the same task."""
    names = _task_names(task)
    destination = Path(run_dir)
    if not destination.is_dir():
        raise FileNotFoundError(
            f"evaluation run directory does not exist: {destination}"
        )
    sidecar = destination / names.directory
    sidecar.mkdir(mode=448, exist_ok=True)
    lock_path = sidecar / ".evaluation.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 384)
    handle = os.fdopen(descriptor, "r+b")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another {task} evaluation is already active for {destination}"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def sidecar_artifact_paths(
    run_dir: str | Path, *, task: str
) -> tuple[Path, Path, Path]:
    """Return deterministic summary, prediction, and identity paths."""
    names = _task_names(task)
    _, final_path, predictions_path, identity_path = _artifact_paths(
        Path(run_dir), names
    )
    return (final_path, predictions_path, identity_path)


def _publish_create_only(path: Path, payload: bytes) -> bool:
    """Create one immutable file, or validate exact bytes already published."""
    if path.exists():
        _require_matching_bytes(path, payload)
        return True
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _require_matching_bytes(path, payload)
            return True
        return False
    finally:
        temporary.unlink(missing_ok=True)


def _require_matching_bytes(path: Path, expected: bytes) -> None:
    if path.read_bytes() != expected:
        raise ValueError(f"new sidecar content conflicts with existing file: {path}")


def _content_digests(
    value: object, *, task: str, identity_path: Path
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {
        "predictions_sha256",
        "summary_sha256",
    }:
        raise ValueError(
            f"existing {task} sidecar identity has invalid published content: {identity_path}"
        )
    digests: dict[str, str] = {}
    for field in ("predictions_sha256", "summary_sha256"):
        digest = value.get(field)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any((character not in "0123456789abcdef" for character in digest))
        ):
            raise ValueError(
                f"existing {task} sidecar identity has invalid {field}: {identity_path}"
            )
        digests[field] = digest
    return digests


def _require_digest(path: Path, expected: str, *, label: str) -> None:
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"{label} cannot be read: {path}") from exc
    if actual != expected:
        raise ValueError(f"{label} digest conflicts with its published identity")


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("sidecar content must be finite JSON data") from exc


def _task_names(task: str) -> _ArtifactNames:
    names = _TASK_NAMES.get(task)
    if names is None:
        raise ValueError(
            f"supported task must be one of {', '.join(sorted(_TASK_NAMES))}"
        )
    return names


def _artifact_paths(
    run_dir: Path, names: _ArtifactNames
) -> tuple[Path, Path, Path, Path]:
    sidecar = run_dir / names.directory
    return (
        sidecar,
        sidecar / names.summary,
        sidecar / names.predictions,
        sidecar / names.identity,
    )


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return dict(value)


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"{label} cannot be read: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} line {line_number} is not valid JSON") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} line {line_number} must be a JSON object")
        rows.append(dict(value))
    return rows
