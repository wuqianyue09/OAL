"""Dependency-light ARC-E protocol and local bundle contracts."""

from __future__ import annotations
from collections.abc import Callable, Iterable, Mapping, Sequence
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import tempfile

ARC_EASY_DATASET_NAME = "allenai/ai2_arc"
ARC_EASY_CONFIG_NAME = "ARC-Easy"
ARC_EASY_SPLIT = "test"
ARC_EASY_TEST_COUNT = 2376
ARC_EASY_PROTOCOL = "arc-easy-harness-0shot-char-v1"
ARC_EASY_BUNDLE_SCHEMA = "qwen_lora_arc_easy_bundle_v1"
ARC_EASY_BUNDLE_DIRECTORY = "allenai_arc_easy_test_v1"
ARC_EASY_ROWS_FILENAME = "test.jsonl"
ARC_EASY_MANIFEST_FILENAME = "manifest.json"
ARC_EASY_HARNESS_REFERENCE_COMMIT = "d6de81643928d653435c431bae19945d41d32520"
_REQUIRED_SOURCE_FIELDS = ("id", "question", "choices", "answerKey")
ArcEasySourceLoader = Callable[[str, str | None], Iterable[Mapping[str, object]]]


def normalize_arc_easy_rows(
    rows: Iterable[Mapping[str, object]], *, expected_count: int = ARC_EASY_TEST_COUNT
) -> list[dict[str, object]]:
    """Preserve the official test order and map answer labels to choice indices."""
    if type(expected_count) is not int or expected_count <= 0:
        raise ValueError("expected_count must be a positive integer")
    normalized: list[dict[str, object]] = []
    observed_ids: set[str] = set()
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"ARC-E row {row_index} must be a mapping")
        missing = [field for field in _REQUIRED_SOURCE_FIELDS if field not in row]
        if missing:
            raise ValueError(
                f"ARC-E row {row_index} is missing required field(s): {', '.join(missing)}"
            )
        identifier = _nonempty_string(row["id"], f"ARC-E row {row_index} id")
        if identifier in observed_ids:
            raise ValueError("ARC-E IDs must be unique")
        observed_ids.add(identifier)
        question = _nonempty_string(row["question"], f"ARC-E row {row_index} question")
        answer_key = _nonempty_string(
            row["answerKey"], f"ARC-E row {row_index} answerKey"
        )
        choices = row["choices"]
        if not isinstance(choices, Mapping):
            raise TypeError(f"ARC-E row {row_index} choices must be a mapping")
        if "text" not in choices or "label" not in choices:
            raise ValueError(
                f"ARC-E row {row_index} choices must contain text and label"
            )
        texts = _string_sequence(choices["text"], f"ARC-E row {row_index} choice text")
        labels = _string_sequence(
            choices["label"], f"ARC-E row {row_index} choice label"
        )
        if len(texts) != len(labels):
            raise ValueError(
                f"ARC-E row {row_index} choice text and label lengths differ"
            )
        if len(set(labels)) != len(labels):
            raise ValueError(f"ARC-E row {row_index} choice labels must be unique")
        if answer_key not in labels:
            raise ValueError(
                f"ARC-E row {row_index} answerKey {answer_key!r} is not present in labels"
            )
        normalized.append(
            {
                "id": identifier,
                "row_index": row_index,
                "question": question,
                "choices": {"text": texts, "label": labels},
                "answerKey": answer_key,
                "label": labels.index(answer_key),
            }
        )
    if len(normalized) != expected_count:
        raise ValueError(
            f"ARC-E test split requires exactly {expected_count} rows; observed {len(normalized)}"
        )
    return normalized


def arc_easy_bundle_path(data_root: str | Path) -> Path:
    return Path(data_root) / "arc_easy" / ARC_EASY_BUNDLE_DIRECTORY


def load_arc_easy_source(
    split: str, requested_source_revision: str | None
) -> Iterable[Mapping[str, object]]:
    """Load the official test split only when preparation requests it."""
    if split != ARC_EASY_SPLIT:
        raise ValueError(f"ARC-E split must be {ARC_EASY_SPLIT!r}")
    datasets = _datasets_module()
    kwargs: dict[str, object] = {}
    if requested_source_revision is not None:
        kwargs["revision"] = requested_source_revision
    loader = getattr(datasets, "load_dataset", None)
    if not callable(loader):
        raise RuntimeError("installed datasets package does not provide load_dataset")
    return loader(
        ARC_EASY_DATASET_NAME, ARC_EASY_CONFIG_NAME, split=ARC_EASY_SPLIT, **kwargs
    )


def installed_datasets_version() -> str:
    version = getattr(_datasets_module(), "__version__", None)
    if not isinstance(version, str) or not version:
        raise RuntimeError("installed datasets package does not expose a version")
    return version


def prepare_arc_easy_bundle(
    data_root: str | Path,
    *,
    source_loader: ArcEasySourceLoader | None,
    tokenizer_identity: Mapping[str, object],
    datasets_version: str | None = None,
    requested_source_revision: str | None = None,
    verify_only: bool = False,
    expected_count: int = ARC_EASY_TEST_COUNT,
) -> dict[str, object]:
    """Create once or validate the canonical offline ARC-E bundle."""
    _validate_revision(requested_source_revision)
    tokenizer = _json_object(tokenizer_identity, "tokenizer_identity")
    if not tokenizer:
        raise ValueError("tokenizer_identity must not be empty")
    destination = arc_easy_bundle_path(data_root)
    if destination.exists():
        manifest = validate_arc_easy_bundle_directory(
            destination,
            tokenizer_identity=tokenizer,
            datasets_version=datasets_version,
            requested_source_revision=requested_source_revision,
            expected_count=expected_count,
        )
        return _bundle_result(
            destination, manifest, expected_count=expected_count, reused=True
        )
    if verify_only:
        raise FileNotFoundError(
            f"ARC-E bundle does not exist for verify-only: {destination}"
        )
    if source_loader is None:
        raise ValueError("source_loader is required to create a missing ARC-E bundle")
    if datasets_version is None:
        datasets_version = installed_datasets_version()
    if not isinstance(datasets_version, str) or not datasets_version:
        raise ValueError("datasets_version must be a non-empty string")
    source = source_loader(ARC_EASY_SPLIT, requested_source_revision)
    fingerprint = getattr(source, "_fingerprint", None)
    if fingerprint is not None and (not isinstance(fingerprint, str)):
        fingerprint = None
    rows = normalize_arc_easy_rows(source, expected_count=expected_count)
    row_bytes = _jsonl_bytes(rows)
    manifest = {
        "schema": ARC_EASY_BUNDLE_SCHEMA,
        "dataset_name": ARC_EASY_DATASET_NAME,
        "dataset_config": ARC_EASY_CONFIG_NAME,
        "split": ARC_EASY_SPLIT,
        "protocol": ARC_EASY_PROTOCOL,
        "harness_reference_commit": ARC_EASY_HARNESS_REFERENCE_COMMIT,
        "requested_source_revision": requested_source_revision,
        "requested_source_revision_kind": "unverified_remote_selector",
        "source_fingerprint": fingerprint,
        "datasets_version": datasets_version,
        "tokenizer": tokenizer,
        "test_count": len(rows),
        "files": {ARC_EASY_ROWS_FILENAME: _file_record(row_bytes)},
        "bundle_sha256": _bundle_sha256(row_bytes),
    }
    _publish_bundle(destination, row_bytes, manifest)
    verified = validate_arc_easy_bundle_directory(
        destination,
        tokenizer_identity=tokenizer,
        datasets_version=datasets_version,
        requested_source_revision=requested_source_revision,
        expected_count=expected_count,
    )
    return _bundle_result(
        destination, verified, expected_count=expected_count, reused=False
    )


def validate_arc_easy_bundle_directory(
    bundle_path: str | Path,
    *,
    tokenizer_identity: Mapping[str, object] | None = None,
    datasets_version: str | None = None,
    requested_source_revision: str | None = None,
    expected_count: int = ARC_EASY_TEST_COUNT,
) -> dict[str, object]:
    root = Path(bundle_path)
    rows_path = root / ARC_EASY_ROWS_FILENAME
    manifest_path = root / ARC_EASY_MANIFEST_FILENAME
    if not root.is_dir():
        raise FileNotFoundError(f"ARC-E bundle directory does not exist: {root}")
    missing = [path.name for path in (rows_path, manifest_path) if not path.is_file()]
    if missing:
        raise ValueError(f"ARC-E bundle is incomplete; missing {', '.join(missing)}")
    try:
        manifest = _json_object(
            json.loads(manifest_path.read_text(encoding="utf-8")), "ARC-E manifest"
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"ARC-E manifest is invalid JSON: {exc.msg}") from exc
    _validate_manifest(manifest, expected_count=expected_count)
    if tokenizer_identity is not None and manifest["tokenizer"] != _json_object(
        tokenizer_identity, "tokenizer_identity"
    ):
        raise ValueError("ARC-E manifest tokenizer identity does not match the request")
    if (
        datasets_version is not None
        and manifest["datasets_version"] != datasets_version
    ):
        raise ValueError("ARC-E manifest datasets_version does not match the request")
    if (
        requested_source_revision is not None
        and manifest["requested_source_revision"] != requested_source_revision
    ):
        raise ValueError(
            "ARC-E manifest requested source revision does not match the request"
        )
    payload = rows_path.read_bytes()
    files = _json_object(manifest["files"], "ARC-E manifest files")
    record = _json_object(files[ARC_EASY_ROWS_FILENAME], "ARC-E file record")
    if len(payload) != record["bytes"]:
        raise ValueError("ARC-E test.jsonl byte count does not match its manifest")
    if hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError("ARC-E test.jsonl sha256 does not match its manifest")
    rows = _validate_prepared_rows(
        _read_jsonl(rows_path), expected_count=expected_count
    )
    if payload != _jsonl_bytes(rows):
        raise ValueError("ARC-E test rows are not canonical JSONL")
    if _bundle_sha256(payload) != manifest["bundle_sha256"]:
        raise ValueError("ARC-E bundle_sha256 does not match local rows")
    return manifest


def load_arc_easy_bundle(
    bundle_path: str | Path, *, expected_count: int = ARC_EASY_TEST_COUNT
) -> tuple[list[dict[str, object]], dict[str, object]]:
    root = Path(bundle_path)
    manifest = validate_arc_easy_bundle_directory(root, expected_count=expected_count)
    rows = _validate_prepared_rows(
        _read_jsonl(root / ARC_EASY_ROWS_FILENAME), expected_count=expected_count
    )
    return (rows, manifest)


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty")
    return value


def _string_sequence(value: object, label: str) -> list[str]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or (not value)
    ):
        raise ValueError(f"{label} must be a non-empty sequence")
    result: list[str] = []
    for index, item in enumerate(value):
        result.append(_nonempty_string(item, f"{label} {index}"))
    return result


def _validate_prepared_rows(
    rows: Iterable[Mapping[str, object]], *, expected_count: int
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    observed_ids: set[str] = set()
    required = {"id", "row_index", "question", "choices", "answerKey", "label"}
    for row_index, row in enumerate(rows):
        value = _json_object(row, f"ARC-E prepared row {row_index}")
        if set(value) != required:
            raise ValueError(f"ARC-E prepared row {row_index} fields are invalid")
        identifier = _nonempty_string(value["id"], f"ARC-E prepared row {row_index} id")
        if identifier in observed_ids:
            raise ValueError("ARC-E prepared row IDs must be unique")
        observed_ids.add(identifier)
        if value["row_index"] != row_index:
            raise ValueError(f"ARC-E prepared row {row_index} index is not canonical")
        question = _nonempty_string(
            value["question"], f"ARC-E prepared row {row_index} question"
        )
        answer_key = _nonempty_string(
            value["answerKey"], f"ARC-E prepared row {row_index} answerKey"
        )
        choices = _json_object(
            value["choices"], f"ARC-E prepared row {row_index} choices"
        )
        if set(choices) != {"text", "label"}:
            raise ValueError(
                f"ARC-E prepared row {row_index} choice fields are invalid"
            )
        texts = _string_sequence(
            choices["text"], f"ARC-E prepared row {row_index} choice text"
        )
        labels = _string_sequence(
            choices["label"], f"ARC-E prepared row {row_index} choice label"
        )
        if len(texts) != len(labels):
            raise ValueError(f"ARC-E prepared row {row_index} choice lengths differ")
        if len(set(labels)) != len(labels):
            raise ValueError(
                f"ARC-E prepared row {row_index} choice labels must be unique"
            )
        if answer_key not in labels:
            raise ValueError(f"ARC-E prepared row {row_index} answerKey is invalid")
        label = value["label"]
        if type(label) is not int or label != labels.index(answer_key):
            raise ValueError(f"ARC-E prepared row {row_index} label is invalid")
        normalized.append(
            {
                "id": identifier,
                "row_index": row_index,
                "question": question,
                "choices": {"text": texts, "label": labels},
                "answerKey": answer_key,
                "label": label,
            }
        )
    if len(normalized) != expected_count:
        raise ValueError(
            f"ARC-E test split requires exactly {expected_count} rows; observed {len(normalized)}"
        )
    return normalized


def _validate_manifest(manifest: dict[str, object], *, expected_count: int) -> None:
    required = {
        "schema",
        "dataset_name",
        "dataset_config",
        "split",
        "protocol",
        "harness_reference_commit",
        "requested_source_revision",
        "requested_source_revision_kind",
        "source_fingerprint",
        "datasets_version",
        "tokenizer",
        "test_count",
        "files",
        "bundle_sha256",
    }
    if set(manifest) != required:
        raise ValueError("ARC-E manifest fields do not match the bundle schema")
    if (
        manifest["schema"] != ARC_EASY_BUNDLE_SCHEMA
        or manifest["dataset_name"] != ARC_EASY_DATASET_NAME
        or manifest["dataset_config"] != ARC_EASY_CONFIG_NAME
        or (manifest["split"] != ARC_EASY_SPLIT)
        or (manifest["protocol"] != ARC_EASY_PROTOCOL)
        or (manifest["harness_reference_commit"] != ARC_EASY_HARNESS_REFERENCE_COMMIT)
    ):
        raise ValueError("ARC-E manifest protocol identity is invalid")
    if manifest["requested_source_revision_kind"] != "unverified_remote_selector":
        raise ValueError("ARC-E manifest revision kind is invalid")
    _validate_revision(manifest["requested_source_revision"])
    if manifest["source_fingerprint"] is not None and (
        not isinstance(manifest["source_fingerprint"], str)
    ):
        raise ValueError("ARC-E manifest source fingerprint is invalid")
    if (
        not isinstance(manifest["datasets_version"], str)
        or not manifest["datasets_version"]
    ):
        raise ValueError("ARC-E manifest datasets_version is invalid")
    if not isinstance(manifest["tokenizer"], Mapping) or not manifest["tokenizer"]:
        raise ValueError("ARC-E manifest tokenizer identity is invalid")
    if manifest["test_count"] != expected_count:
        raise ValueError("ARC-E manifest test count is invalid")
    files = _json_object(manifest["files"], "ARC-E manifest files")
    if set(files) != {ARC_EASY_ROWS_FILENAME}:
        raise ValueError("ARC-E manifest files are invalid")
    record = _json_object(files[ARC_EASY_ROWS_FILENAME], "ARC-E file record")
    if set(record) != {"bytes", "sha256"} or type(record["bytes"]) is not int:
        raise ValueError("ARC-E test file record is invalid")
    if not _is_sha256(record["sha256"]) or not _is_sha256(manifest["bundle_sha256"]):
        raise ValueError("ARC-E manifest sha256 fields are invalid")


def _publish_bundle(
    destination: Path, rows: bytes, manifest: Mapping[str, object]
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    published = False
    try:
        (temporary / ARC_EASY_ROWS_FILENAME).write_bytes(rows)
        (temporary / ARC_EASY_MANIFEST_FILENAME).write_bytes(
            _canonical_json_bytes(manifest)
        )
        for path in temporary.iterdir():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        os.symlink(temporary.name, destination, target_is_directory=True)
        published = True
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    if text.endswith("\n"):
        text = text[:-1]
    if not text:
        return []
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(text.split("\n"), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"ARC-E {path.name} line {line_number} is invalid JSON"
            ) from exc
        rows.append(_json_object(value, f"ARC-E {path.name} line {line_number}"))
    return rows


def _json_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    try:
        copied = json.loads(_canonical_json_bytes(dict(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON-serializable") from exc
    if not isinstance(copied, dict):
        raise AssertionError("canonical JSON unexpectedly changed object type")
    return copied


def _jsonl_bytes(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join((_canonical_json_bytes(row) + b"\n" for row in rows))


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _file_record(payload: bytes) -> dict[str, object]:
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _bundle_sha256(payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(ARC_EASY_ROWS_FILENAME.encode("ascii"))
    digest.update(b"\x00")
    digest.update(payload)
    return digest.hexdigest()


def _bundle_result(
    destination: Path,
    manifest: Mapping[str, object],
    *,
    expected_count: int,
    reused: bool,
) -> dict[str, object]:
    return {
        "bundle_path": str(destination),
        "bundle_sha256": manifest["bundle_sha256"],
        "test_count": expected_count,
        "reused": reused,
    }


def _datasets_module() -> object:
    try:
        return importlib.import_module("datasets")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ARC-E preparation requires the optional data dependency; install qwen-lora-experiment[data]"
        ) from exc


def _validate_revision(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError("requested_source_revision must be None or a non-empty string")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all((character in "0123456789abcdef" for character in value))
    )
