"""Immutable evaluation records and reusable prediction publication."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import json
import os
from pathlib import Path
import stat
import tempfile
from uuid import uuid4
from .checkpointing import CheckpointContext
from .evaluation_contracts import (
    EvaluationContext,
    MMLU_SIDECAR_DIRECTORY,
    MmluScore,
    NLL_EVALUATION_KIND,
    PiqaScore,
    _json_object,
    _mmlu_contract,
)
from .paths import SCHEMA_VERSION, canonical_json, fsync_directory, sha256_bytes


def _create_or_validate_immutable_stage_record(
    path: Path, record: Mapping[str, object], label: str
) -> None:
    """Publish one final endpoint record with no overwrite race."""
    if not path.parent.is_dir():
        raise FileNotFoundError(f"{label} output parent does not exist: {path.parent}")
    encoded = (canonical_json(_json_object(record, f"{label} record")) + "\n").encode(
        "utf-8"
    )
    from .run_artifacts import _create_or_validate_immutable_bytes

    _create_or_validate_immutable_bytes(path, encoded, label)


def _load_nll_stage_record(path: Path) -> dict[str, object]:
    try:
        record = json.loads(_safe_regular_file_bytes(path, "NLL evaluation stage"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"NLL evaluation stage is invalid JSON: {path}") from exc
    if not isinstance(record, dict) or record.get("kind") != NLL_EVALUATION_KIND:
        raise ValueError("NLL evaluation stage has the wrong kind")
    return record


def _piqa_prediction_identity(
    *,
    checkpoint_context: EvaluationContext,
    evaluation_source: Mapping[str, object],
    execution: Mapping[str, object],
    data_identity: Mapping[str, object],
    model_identity: Mapping[str, object],
    experiment_identity: Mapping[str, object],
    normalized_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return the stable inputs that uniquely authorize PIQA prediction reuse."""
    rows = [
        _json_object(row, f"PIQA input row {index}")
        for (index, row) in enumerate(normalized_rows)
    ]
    identity = {
        "checkpoint_context": {
            "config_identity": _json_object(
                checkpoint_context.config_identity, "checkpoint_context.config_identity"
            ),
            "data_identity": _json_object(
                checkpoint_context.data_identity, "checkpoint_context.data_identity"
            ),
            "model_identity": _json_object(
                checkpoint_context.model_identity, "checkpoint_context.model_identity"
            ),
        },
        "execution": _json_object(execution, "execution"),
        "data_identity": _json_object(data_identity, "data_identity"),
        "model_identity": _json_object(model_identity, "model_identity"),
        "experiment_identity": _json_object(experiment_identity, "experiment_identity"),
        "piqa_input_count": len(rows),
        "piqa_input_sha256": sha256_bytes(canonical_json(rows).encode("utf-8")),
    }
    normalized_source = _json_object(evaluation_source, "evaluation_source")
    identity["selected_checkpoint_sha256"] = normalized_source.get("sha256")
    return identity


def _normalize_mmlu_data_identity(value: Mapping[str, object]) -> dict[str, object]:
    identity = _json_object(value, "mmlu_data_identity")
    digest = identity.get("bundle_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any((character not in "0123456789abcdef" for character in digest))
    ):
        raise ValueError("mmlu_data_identity.bundle_sha256 must be a lowercase SHA-256")
    tokenizer = identity.get("tokenizer")
    if not isinstance(tokenizer, Mapping) or not _json_object(
        tokenizer, "mmlu_data_identity.tokenizer"
    ):
        raise ValueError("mmlu_data_identity.tokenizer must be a non-empty mapping")
    return identity


def _mmlu_prediction_identity(
    *,
    checkpoint_context: CheckpointContext,
    evaluation_source: Mapping[str, object],
    execution: Mapping[str, object],
    data_identity: Mapping[str, object],
    model_identity: Mapping[str, object],
    experiment_identity: Mapping[str, object],
    mmlu_data_identity: Mapping[str, object],
    dev_rows: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
    expected_test_ids: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    """Bind sidecar predictions to the adapter and immutable MMLU inputs."""
    mmlu = _mmlu_contract()
    normalized_dev = [
        _json_object(row, f"MMLU dev row {index}")
        for (index, row) in enumerate(dev_rows)
    ]
    normalized_test = [
        _json_object(row, f"MMLU test row {index}")
        for (index, row) in enumerate(test_rows)
    ]
    expected = {
        subject: list(identifiers)
        for (subject, identifiers) in sorted(expected_test_ids.items())
    }
    return {
        "checkpoint_context": {
            "config_identity": _json_object(
                checkpoint_context.config_identity, "checkpoint_context.config_identity"
            ),
            "data_identity": _json_object(
                checkpoint_context.data_identity, "checkpoint_context.data_identity"
            ),
            "model_identity": _json_object(
                checkpoint_context.model_identity, "checkpoint_context.model_identity"
            ),
        },
        "selected_checkpoint_sha256": _json_object(
            evaluation_source, "evaluation_source"
        ).get("sha256"),
        "execution": _json_object(execution, "execution"),
        "data_identity": _json_object(data_identity, "data_identity"),
        "model_identity": _json_object(model_identity, "model_identity"),
        "experiment_identity": _json_object(experiment_identity, "experiment_identity"),
        "mmlu_protocol": mmlu.MMLU_PROTOCOL,
        "mmlu_data_identity": _normalize_mmlu_data_identity(mmlu_data_identity),
        "mmlu_dev_input_sha256": sha256_bytes(
            canonical_json(normalized_dev).encode("utf-8")
        ),
        "mmlu_test_input_sha256": sha256_bytes(
            canonical_json(normalized_test).encode("utf-8")
        ),
        "expected_test_ids": expected,
    }


def _mmlu_prediction_identity_record(
    identity: Mapping[str, object], *, encoded_predictions: bytes
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "qwen_lora_mmlu_predictions_identity",
        "prediction_identity": _json_object(identity, "MMLU prediction identity"),
        "prediction_sha256": sha256_bytes(encoded_predictions),
    }


@contextmanager
def _open_mmlu_sidecar_directory(destination: Path):
    """Yield a stable no-follow descriptor for the MMLU output namespace.

    MMLU is deliberately a sidecar, so it must never be able to publish into
    a directory reached by replacing ``mmlu_sidecar`` with a symbolic link
    while scoring is in progress.  All sidecar I/O therefore uses this opened
    directory descriptor rather than re-resolving the pathname.
    """
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        destination_descriptor = os.open(destination, directory_flags)
    except OSError as exc:
        raise ValueError(
            "MMLU evaluation run directory must be a non-symlink directory"
        ) from exc
    try:
        try:
            sidecar_stat = os.stat(
                MMLU_SIDECAR_DIRECTORY,
                dir_fd=destination_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            try:
                os.mkdir(
                    MMLU_SIDECAR_DIRECTORY, mode=448, dir_fd=destination_descriptor
                )
            except FileExistsError:
                pass
            sidecar_stat = os.stat(
                MMLU_SIDECAR_DIRECTORY,
                dir_fd=destination_descriptor,
                follow_symlinks=False,
            )
        if stat.S_ISLNK(sidecar_stat.st_mode) or not stat.S_ISDIR(sidecar_stat.st_mode):
            raise ValueError("MMLU sidecar namespace must be a non-symlink directory")
        try:
            sidecar_descriptor = os.open(
                MMLU_SIDECAR_DIRECTORY, directory_flags, dir_fd=destination_descriptor
            )
        except OSError as exc:
            raise ValueError("MMLU sidecar namespace changed during open") from exc
        try:
            opened_stat = os.fstat(sidecar_descriptor)
            if not stat.S_ISDIR(opened_stat.st_mode) or (
                opened_stat.st_dev,
                opened_stat.st_ino,
            ) != (sidecar_stat.st_dev, sidecar_stat.st_ino):
                raise ValueError("MMLU sidecar namespace changed during open")
            yield sidecar_descriptor
        finally:
            os.close(sidecar_descriptor)
    finally:
        os.close(destination_descriptor)


def _mmlu_sidecar_namespace_is_current(
    destination: Path, sidecar_descriptor: int
) -> bool:
    """Return whether the visible namespace still names the opened directory."""
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        destination_descriptor = os.open(destination, directory_flags)
    except OSError:
        return False
    try:
        try:
            current = os.stat(
                MMLU_SIDECAR_DIRECTORY,
                dir_fd=destination_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return False
        opened = os.fstat(sidecar_descriptor)
        return (
            stat.S_ISDIR(current.st_mode)
            and (not stat.S_ISLNK(current.st_mode))
            and ((current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino))
        )
    finally:
        os.close(destination_descriptor)


def _sidecar_regular_file_exists(
    directory_descriptor: int, name: str, label: str
) -> bool:
    try:
        entry = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    return True


def _safe_regular_file_bytes_at(
    directory_descriptor: int, name: str, label: str
) -> bytes:
    """Read a stable retained sidecar file without resolving its pathname."""
    try:
        before = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} does not exist: {name}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
    except OSError as exc:
        raise ValueError(f"{label} changed during safe open") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise ValueError(f"{label} changed during safe open")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    try:
        after = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ValueError(f"{label} changed during safe read") from exc
    if (
        stat.S_ISLNK(after.st_mode)
        or not stat.S_ISREG(after.st_mode)
        or (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        raise ValueError(f"{label} changed during safe read")
    return b"".join(chunks)


def _create_or_validate_immutable_sidecar_record(
    directory_descriptor: int, name: str, record: Mapping[str, object], label: str
) -> None:
    encoded = (canonical_json(_json_object(record, f"{label} record")) + "\n").encode(
        "utf-8"
    )
    _create_or_validate_immutable_sidecar_bytes(
        directory_descriptor, name, encoded, label
    )


def _create_or_validate_immutable_sidecar_bytes(
    directory_descriptor: int, name: str, encoded: bytes, label: str
) -> None:
    """Create one immutable sidecar file through the stable directory FD."""
    try:
        existing = _safe_regular_file_bytes_at(directory_descriptor, name, label)
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if existing != encoded:
            raise ValueError(f"{label} already exists with different immutable bytes")
        return
    temporary_name = f".{name}.{uuid4().hex}.tmp"
    descriptor = os.open(
        temporary_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        384,
        dir_fd=directory_descriptor,
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(
                temporary_name,
                name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            pass
        os.fsync(directory_descriptor)
    finally:
        try:
            os.unlink(temporary_name, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass
    existing = _safe_regular_file_bytes_at(directory_descriptor, name, label)
    if existing != encoded:
        raise ValueError(f"{label} already exists with different immutable bytes")


def _write_or_validate_sidecar_predictions(
    directory_descriptor: int,
    name: str,
    *,
    encoded_predictions: bytes,
    expected_identity: Mapping[str, object],
    label: str,
) -> None:
    """Publish deterministic sidecar JSONL through the stable directory FD."""
    try:
        existing = _safe_regular_file_bytes_at(
            directory_descriptor, name, f"{label} predictions"
        )
    except FileNotFoundError:
        existing = None
    if existing is None:
        _create_or_validate_immutable_sidecar_bytes(
            directory_descriptor, name, encoded_predictions, f"{label} predictions"
        )
        existing = _safe_regular_file_bytes_at(
            directory_descriptor, name, f"{label} predictions"
        )
    if sha256_bytes(existing) != expected_identity["prediction_sha256"]:
        raise ValueError(
            f"{label} predictions already exist with a different immutable digest"
        )


def _load_reusable_mmlu_predictions_at(
    *,
    directory_descriptor: int,
    predictions_name: str,
    identity_name: str,
    expected_identity: Mapping[str, object],
) -> MmluScore | None:
    predictions_exists = _sidecar_regular_file_exists(
        directory_descriptor, predictions_name, "MMLU predictions"
    )
    identity_exists = _sidecar_regular_file_exists(
        directory_descriptor, identity_name, "MMLU prediction identity"
    )
    if not predictions_exists and (not identity_exists):
        return None
    if not predictions_exists:
        _validate_mmlu_prediction_identity_record(
            _load_mmlu_prediction_identity_record_at(
                directory_descriptor, identity_name
            ),
            expected_identity,
        )
        return None
    if not identity_exists:
        raise ValueError("MMLU predictions exist without an immutable identity sidecar")
    record = _load_mmlu_prediction_identity_record_at(
        directory_descriptor, identity_name
    )
    _validate_mmlu_prediction_identity_record(record, expected_identity)
    encoded = _safe_regular_file_bytes_at(
        directory_descriptor, predictions_name, "MMLU predictions"
    )
    if sha256_bytes(encoded) != record["prediction_sha256"]:
        raise ValueError(
            "MMLU predictions do not match their immutable identity digest"
        )
    return _mmlu_score_from_prediction_bytes(encoded)


def _load_mmlu_prediction_identity_record_at(
    directory_descriptor: int, name: str
) -> dict[str, object]:
    encoded = _safe_regular_file_bytes_at(
        directory_descriptor, name, "MMLU prediction identity"
    )
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("MMLU prediction identity is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("MMLU prediction identity must be a JSON object")
    return dict(value)


def _validate_mmlu_prediction_identity_record(
    record: Mapping[str, object], expected_identity: Mapping[str, object]
) -> None:
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("kind") != "qwen_lora_mmlu_predictions_identity"
        or record.get("prediction_identity")
        != _json_object(expected_identity, "MMLU prediction identity")
    ):
        raise ValueError(
            "MMLU prediction identity does not match the current evaluation inputs"
        )
    digest = record.get("prediction_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any((character not in "0123456789abcdef" for character in digest))
    ):
        raise ValueError("MMLU prediction identity has an invalid prediction SHA-256")


def _piqa_prediction_identity_record(
    identity: Mapping[str, object], *, encoded_predictions: bytes
) -> dict[str, object]:
    """Bind one fully encoded JSONL file to immutable scoring inputs."""
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "qwen_lora_piqa_predictions_identity",
        "prediction_identity": _json_object(identity, "PIQA prediction identity"),
        "prediction_sha256": sha256_bytes(encoded_predictions),
    }


def _encode_prediction_records(
    predictions: Sequence[Mapping[str, object]], *, label: str = "PIQA"
) -> bytes:
    """Canonicalize all predictions before any final path is made visible."""
    records = [
        _json_object(prediction, f"{label} prediction {index}")
        for (index, prediction) in enumerate(predictions)
    ]
    if not records:
        raise ValueError(f"{label} prediction publication requires at least one record")
    return "".join((canonical_json(record) + "\n" for record in records)).encode(
        "utf-8"
    )


def _load_reusable_piqa_predictions(
    *,
    predictions_path: Path,
    identity_path: Path,
    expected_identity: Mapping[str, object],
) -> PiqaScore | None:
    """Load a retained prediction file only when its immutable sidecar matches."""
    predictions_exists = predictions_path.exists()
    identity_exists = identity_path.exists()
    if not predictions_exists and (not identity_exists):
        return None
    if not predictions_exists:
        _validate_prediction_identity_record(
            _load_prediction_identity_record(identity_path), expected_identity
        )
        return None
    if not identity_exists:
        raise ValueError("PIQA predictions exist without an immutable identity sidecar")
    record = _load_prediction_identity_record(identity_path)
    _validate_prediction_identity_record(record, expected_identity)
    encoded = _safe_regular_file_bytes(predictions_path, "PIQA predictions")
    expected_digest = record["prediction_sha256"]
    if sha256_bytes(encoded) != expected_digest:
        raise ValueError(
            "PIQA predictions do not match their immutable identity digest"
        )
    return _piqa_score_from_prediction_bytes(encoded)


def _load_prediction_identity_record(path: Path) -> dict[str, object]:
    encoded = _safe_regular_file_bytes(path, "PIQA prediction identity")
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("PIQA prediction identity is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("PIQA prediction identity must be a JSON object")
    return dict(value)


def _validate_prediction_identity_record(
    record: Mapping[str, object], expected_identity: Mapping[str, object]
) -> None:
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("kind") != "qwen_lora_piqa_predictions_identity"
        or record.get("prediction_identity")
        != _json_object(expected_identity, "PIQA prediction identity")
    ):
        raise ValueError(
            "PIQA prediction identity does not match the current evaluation inputs"
        )
    digest = record.get("prediction_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any((character not in "0123456789abcdef" for character in digest))
    ):
        raise ValueError("PIQA prediction identity has an invalid prediction SHA-256")


def _write_or_validate_prediction_records(
    path: Path, *, encoded_predictions: bytes, expected_identity: Mapping[str, object]
) -> None:
    """No-clobber publish deterministic JSONL, or validate a retained retry file."""
    if path.exists():
        existing = _safe_regular_file_bytes(path, "PIQA predictions")
        if sha256_bytes(existing) != expected_identity["prediction_sha256"]:
            raise ValueError(
                "PIQA predictions already exist with a different immutable digest"
            )
        return
    _publish_prediction_bytes(path, encoded_predictions)
    existing = _safe_regular_file_bytes(path, "PIQA predictions")
    if sha256_bytes(existing) != expected_identity["prediction_sha256"]:
        raise ValueError(
            "published PIQA predictions do not match their immutable digest"
        )


def _publish_prediction_bytes(path: Path, encoded: bytes) -> None:
    """Atomically expose a complete fsynced JSONL file without pathname cleanup."""
    if not path.parent.is_dir():
        raise FileNotFoundError(
            f"PIQA predictions output parent does not exist: {path.parent}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
        fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _safe_regular_file_bytes(path: Path, label: str) -> bytes:
    """Read a retained file through a no-follow descriptor and stable inode."""
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} does not exist: {path}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise ValueError(f"{label} changed during safe open")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if (
        stat.S_ISLNK(after.st_mode)
        or not stat.S_ISREG(after.st_mode)
        or (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        raise ValueError(f"{label} changed during safe read")
    return b"".join(chunks)


def _piqa_score_from_prediction_bytes(encoded: bytes) -> PiqaScore:
    """Reconstruct the score summary from a validated retained JSONL payload."""
    try:
        lines = encoded.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("PIQA predictions are not UTF-8 JSONL") from exc
    if not lines:
        raise ValueError("PIQA predictions contain no records")
    records: list[dict[str, object]] = []
    raw_correct = 0
    norm_correct = 0
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"PIQA prediction {index} is invalid JSON") from exc
        if not isinstance(record, Mapping):
            raise ValueError(f"PIQA prediction {index} must be an object")
        normalized = _json_object(record, f"PIQA prediction {index}")
        label = normalized.get("label")
        prediction_norm = normalized.get("prediction_norm")
        raw = normalized.get("raw_correct")
        if (
            type(label) is not int
            or type(prediction_norm) is not int
            or type(raw) is not bool
        ):
            raise ValueError(f"PIQA prediction {index} lacks score evidence")
        records.append(normalized)
        raw_correct += int(raw)
        norm_correct += int(prediction_norm == label)
    count = len(records)
    return PiqaScore(
        predictions=tuple(records),
        raw_accuracy=raw_correct / count,
        acc_norm=norm_correct / count,
    )


def _mmlu_score_from_prediction_bytes(encoded: bytes) -> MmluScore:
    """Reconstruct MMLU aggregates from a validated immutable JSONL file."""
    try:
        lines = encoded.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("MMLU predictions are not UTF-8 JSONL") from exc
    if not lines:
        raise ValueError("MMLU predictions contain no records")
    records: list[dict[str, object]] = []
    by_subject: dict[str, list[bool]] = {}
    correct_count = 0
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"MMLU prediction {index} is invalid JSON") from exc
        normalized = _json_object(record, f"MMLU prediction {index}")
        subject = normalized.get("subject")
        label = normalized.get("label")
        prediction = normalized.get("prediction")
        is_correct = normalized.get("correct")
        if (
            not isinstance(subject, str)
            or type(label) is not int
            or type(prediction) is not int
            or (type(is_correct) is not bool)
            or (label not in range(4))
            or (prediction not in range(4))
            or (is_correct != (prediction == label))
        ):
            raise ValueError(f"MMLU prediction {index} lacks score evidence")
        records.append(normalized)
        by_subject.setdefault(subject, []).append(is_correct)
        correct_count += int(is_correct)
    summaries = tuple(
        (
            {
                "subject": subject,
                "question_count": len(correctness),
                "correct_count": sum(correctness),
                "accuracy": sum(correctness) / len(correctness),
            }
            for (subject, correctness) in sorted(by_subject.items())
        )
    )
    return MmluScore(
        predictions=tuple(records),
        macro_accuracy=sum((float(summary["accuracy"]) for summary in summaries))
        / len(summaries),
        micro_accuracy=correct_count / len(records),
        subject_summaries=summaries,
    )


def _require_new_output_path(path: Path, label: str) -> None:
    if not path.parent.is_dir():
        raise FileNotFoundError(f"{label} output parent does not exist: {path.parent}")
    if path.exists():
        raise FileExistsError(
            f"{label} output already exists and will not be overwritten: {path}"
        )
