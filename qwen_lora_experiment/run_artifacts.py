"""Run artifact names, validated reads, and durable evidence publication."""

from __future__ import annotations
from collections.abc import Callable, Mapping
import io
import json
import os
from pathlib import Path
import stat
import tempfile
from .paths import (
    _is_sha256,
    atomic_write_json,
    canonical_json,
    fsync_directory,
    sha256_bytes,
)
from .telemetry import require_cohort_member_mutable, cohort_member_lock
from .workflows.errors import OrchestrationError

EFFECTIVE_CONFIG_FILENAME = "effective_config.json"
DATA_MANIFEST_COPY_FILENAME = "data_manifest.json"
PREFLIGHT_FILENAME = "preflight.json"
SOURCE_HASHES_FILENAME = "source_hashes.json"
RUNTIME_EVIDENCE_FILENAME = "runtime_evidence.json"
LORA_INITIAL_HASHES_FILENAME = "lora_initial_hashes.json"


def _read_json_mapping(path: Path, label: str) -> dict[str, object]:
    """Read one persisted evidence object with a caller-specific error label."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OrchestrationError(f"{label} is unreadable") from exc
    if not isinstance(value, Mapping):
        raise OrchestrationError(f"{label} must be a JSON object")
    return dict(value)


def _create_or_validate_immutable_bytes(
    path: Path,
    encoded: bytes,
    label: str,
    *,
    before_publish: Callable[[], None] | None = None,
) -> None:
    """Atomically publish one fully-written immutable record without clobbering.

    The temporary inode lives beside its final path.  Linking that completed
    inode into place provides no-clobber publication: a reader sees either no
    final file or the whole fsynced record, never a partially written file.
    A racing publisher is acceptable only when it produced the exact same
    canonical bytes.
    """
    if not isinstance(encoded, bytes):
        raise TypeError("immutable record bytes must be bytes")
    if before_publish is not None and (not callable(before_publish)):
        raise TypeError("before_publish must be callable when provided")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"{label} parent does not exist: {path.parent}")
    if _validate_existing_immutable_bytes(path, encoded, label):
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        while True:
            if before_publish is not None:
                before_publish()
            try:
                os.link(temporary, path)
                break
            except FileExistsError:
                if _validate_existing_immutable_bytes(path, encoded, label):
                    return
        fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _validate_existing_immutable_bytes(path: Path, encoded: bytes, label: str) -> bool:
    """Return true only for a stable regular file with the requested bytes."""
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(path_stat.st_mode):
        raise ValueError(f"{label} immutable destination must not be a symlink")
    if not stat.S_ISREG(path_stat.st_mode):
        raise ValueError(f"{label} immutable destination must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise OrchestrationError(
            f"{label} cannot be opened for immutable comparison"
        ) from exc
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise ValueError(f"{label} immutable destination must be a regular file")
        if (opened_stat.st_dev, opened_stat.st_ino) != (
            path_stat.st_dev,
            path_stat.st_ino,
        ):
            raise OrchestrationError(
                f"{label} immutable destination changed during comparison"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        current = b"".join(chunks)
    finally:
        os.close(descriptor)
    try:
        current_path_stat = path.lstat()
    except FileNotFoundError as exc:
        raise OrchestrationError(
            f"{label} immutable destination changed during comparison"
        ) from exc
    if (
        stat.S_ISLNK(current_path_stat.st_mode)
        or not stat.S_ISREG(current_path_stat.st_mode)
        or (current_path_stat.st_dev, current_path_stat.st_ino)
        != (opened_stat.st_dev, opened_stat.st_ino)
    ):
        raise OrchestrationError(
            f"{label} immutable destination changed during comparison"
        )
    if current != encoded:
        raise ValueError(f"{label} already exists with different immutable contents")
    fsync_directory(path.parent)
    return True


def _create_or_validate_immutable_json(
    path: Path,
    value: Mapping[str, object],
    label: str,
    *,
    before_publish: Callable[[], None] | None = None,
) -> None:
    """Create one canonical JSON record, accepting only identical retries."""
    encoded = (canonical_json(dict(value)) + "\n").encode("utf-8")
    _create_or_validate_immutable_bytes(
        path, encoded, label, before_publish=before_publish
    )


def _validated_source_hash_mapping(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise OrchestrationError(f"{label} must be a mapping")
    normalized: dict[str, str] = {}
    for source_path, digest in value.items():
        if not isinstance(source_path, str) or not source_path:
            raise OrchestrationError(f"{label} keys must be non-empty path strings")
        if not _is_sha256(digest):
            raise OrchestrationError(f"{label} values must be lowercase SHA-256")
        normalized[source_path] = digest
    return normalized


def _write_mutable_run_evidence(
    run_dir: Path, filename: str, value: Mapping[str, object]
) -> None:
    """Write one pre-freeze run-evidence file under the shared member lock."""
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise ValueError("run evidence filename must be one safe path component")
    with cohort_member_lock(run_dir):
        require_cohort_member_mutable(run_dir)
        atomic_write_json(run_dir / filename, value)


def _safe_run_artifact_bytes(run_dir: Path, filename: str) -> bytes:
    """Read run evidence through a stable, no-follow file descriptor.

    Retain the historical formal-cohort error labels used by artifact readers.
    """
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise ValueError(
            "formal cohort source filename must be one safe path component"
        )
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"formal cohort member run directory does not exist: {root}"
        )
    source = root / filename
    try:
        before = source.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"formal cohort member source evidence is missing: {source}"
        ) from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(
            "formal cohort member source evidence must be a regular non-symlink file"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise ValueError(
            "formal cohort member source evidence cannot be opened safely"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise ValueError(
                "formal cohort member source evidence changed during safe open"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    try:
        after = source.lstat()
    except FileNotFoundError as exc:
        raise ValueError(
            "formal cohort member source evidence changed during safe read"
        ) from exc
    if (
        stat.S_ISLNK(after.st_mode)
        or not stat.S_ISREG(after.st_mode)
        or (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        raise ValueError(
            "formal cohort member source evidence changed during safe read"
        )
    return b"".join(chunks)


def _safe_run_artifact_sha256(run_dir: Path, filename: str) -> str:
    return sha256_bytes(_safe_run_artifact_bytes(run_dir, filename))


def _safe_run_artifact_json(
    run_dir: Path, filename: str, label: str
) -> dict[str, object]:
    try:
        value = json.loads(_safe_run_artifact_bytes(run_dir, filename))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"formal cohort {label} is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"formal cohort {label} must be a JSON object")
    return dict(value)


def _load_run_checkpoint_payload_bytes(
    encoded: bytes, path: Path
) -> Mapping[str, object]:
    """Decode one already no-follow-read checkpoint byte snapshot."""
    if not isinstance(encoded, bytes):
        raise TypeError("formal cohort selected checkpoint bytes must be bytes")
    try:
        import torch

        value = torch.load(io.BytesIO(encoded), map_location="cpu", weights_only=True)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"formal cohort selected checkpoint cannot be read: {path}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ValueError("formal cohort selected checkpoint payload must be a mapping")
    return value
