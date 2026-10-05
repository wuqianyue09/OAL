"""Dependency-light HellaSwag protocol and local bundle contracts."""

from __future__ import annotations
from collections.abc import Callable, Iterable, Mapping, Sequence
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

HELLASWAG_DATASET_NAME = "Rowan/hellaswag"
HELLASWAG_SPLIT = "validation"
HELLASWAG_VALIDATION_COUNT = 10042
HELLASWAG_PROTOCOL = "hellaswag-harness-0shot-v2"
HELLASWAG_BUNDLE_SCHEMA = "qwen_lora_hellaswag_bundle_v2"
HELLASWAG_BUNDLE_DIRECTORY = "rowan_hellaswag_validation_v2"
HELLASWAG_ROWS_FILENAME = "validation.jsonl"
HELLASWAG_MANIFEST_FILENAME = "manifest.json"
_BRACKETED_ARTIFACT = re.compile("\\[.*?\\]")
_REQUIRED_SOURCE_FIELDS = (
    "ind",
    "activity_label",
    "ctx_a",
    "ctx_b",
    "endings",
    "label",
)
HellaSwagSourceLoader = Callable[[str, str | None], Iterable[Mapping[str, object]]]


def preprocess_hellaswag_text(text: str) -> str:
    """Apply the public lm-evaluation-harness HellaSwag cleanup."""
    if not isinstance(text, str):
        raise TypeError("HellaSwag text must be a string")
    cleaned = _BRACKETED_ARTIFACT.sub("", text.strip().replace(" [title]", ". "))
    return cleaned.replace("  ", " ")


def normalize_hellaswag_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    expected_count: int = HELLASWAG_VALIDATION_COUNT,
) -> list[dict[str, object]]:
    """Normalize the labeled validation split in canonical row order."""
    if type(expected_count) is not int or expected_count <= 0:
        raise ValueError("expected_count must be a positive integer")
    normalized: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"HellaSwag row {row_index} must be a mapping")
        missing = [field for field in _REQUIRED_SOURCE_FIELDS if field not in row]
        if missing:
            raise ValueError(
                f"HellaSwag row {row_index} is missing required field(s): {', '.join(missing)}"
            )
        activity = row["activity_label"]
        ctx_a = row["ctx_a"]
        ctx_b = row["ctx_b"]
        if not all((isinstance(value, str) for value in (activity, ctx_a, ctx_b))):
            raise TypeError(f"HellaSwag row {row_index} context fields must be strings")
        endings = row["endings"]
        if not isinstance(endings, Sequence) or isinstance(endings, (str, bytes)):
            raise TypeError(f"HellaSwag row {row_index}.endings must be a sequence")
        if len(endings) != 4:
            raise ValueError(
                f"HellaSwag row {row_index}.endings must contain exactly four choices"
            )
        continuations: list[str] = []
        for ending_index, ending in enumerate(endings):
            if not isinstance(ending, str):
                raise TypeError(
                    f"HellaSwag row {row_index}.endings[{ending_index}] must be a string"
                )
            cleaned = preprocess_hellaswag_text(ending)
            if not cleaned:
                raise ValueError(
                    f"HellaSwag row {row_index}.endings[{ending_index}] is empty after cleanup"
                )
            continuations.append(" " + cleaned)
        label = row["label"]
        if isinstance(label, str) and label in {"0", "1", "2", "3"}:
            label = int(label)
        if type(label) is not int or label not in range(4):
            raise ValueError(
                f"HellaSwag row {row_index}.label must be an integer from 0 through 3"
            )
        source_index = row["ind"]
        if not isinstance(source_index, (int, str)) or isinstance(source_index, bool):
            raise TypeError(
                f"HellaSwag row {row_index}.ind must be an integer or string"
            )
        context = preprocess_hellaswag_text(f"{activity}: {ctx_a} {ctx_b.capitalize()}")
        if not context:
            raise ValueError(
                f"HellaSwag row {row_index} context is empty after cleanup"
            )
        normalized.append(
            {
                "id": f"validation:{row_index}",
                "source_index": source_index,
                "context": context,
                "continuations": continuations,
                "label": label,
            }
        )
    if len(normalized) != expected_count:
        raise ValueError(
            f"HellaSwag validation split requires exactly {expected_count} rows; observed {len(normalized)}"
        )
    return normalized


def hellaswag_bundle_path(data_root: str | Path) -> Path:
    return Path(data_root) / "hellaswag" / HELLASWAG_BUNDLE_DIRECTORY


def load_hellaswag_source(
    split: str, requested_source_revision: str | None
) -> Iterable[Mapping[str, object]]:
    """Load the official split only when the preparation command requests it."""
    if split != HELLASWAG_SPLIT:
        raise ValueError(f"HellaSwag split must be {HELLASWAG_SPLIT!r}")
    datasets = _datasets_module()
    kwargs: dict[str, object] = {}
    if requested_source_revision is not None:
        kwargs["revision"] = requested_source_revision
    loader = getattr(datasets, "load_dataset", None)
    if not callable(loader):
        raise RuntimeError("installed datasets package does not provide load_dataset")
    return loader(HELLASWAG_DATASET_NAME, split=split, **kwargs)


def installed_datasets_version() -> str:
    version = getattr(_datasets_module(), "__version__", None)
    if not isinstance(version, str) or not version:
        raise RuntimeError("installed datasets package does not expose a version")
    return version


def prepare_hellaswag_bundle(
    data_root: str | Path,
    *,
    source_loader: HellaSwagSourceLoader | None,
    tokenizer_identity: Mapping[str, object],
    datasets_version: str | None = None,
    requested_source_revision: str | None = None,
    verify_only: bool = False,
    expected_count: int = HELLASWAG_VALIDATION_COUNT,
) -> dict[str, object]:
    """Create once or validate the canonical offline HellaSwag bundle."""
    _validate_revision(requested_source_revision)
    tokenizer = _json_object(tokenizer_identity, "tokenizer_identity")
    if not tokenizer:
        raise ValueError("tokenizer_identity must not be empty")
    destination = hellaswag_bundle_path(data_root)
    if destination.exists():
        manifest = validate_hellaswag_bundle_directory(
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
            f"HellaSwag bundle does not exist for verify-only: {destination}"
        )
    if source_loader is None:
        raise ValueError(
            "source_loader is required to create a missing HellaSwag bundle"
        )
    if datasets_version is None:
        datasets_version = installed_datasets_version()
    if not isinstance(datasets_version, str) or not datasets_version:
        raise ValueError("datasets_version must be a non-empty string")
    source = source_loader(HELLASWAG_SPLIT, requested_source_revision)
    fingerprint = getattr(source, "_fingerprint", None)
    if fingerprint is not None and (not isinstance(fingerprint, str)):
        fingerprint = None
    rows = normalize_hellaswag_rows(source, expected_count=expected_count)
    row_bytes = _jsonl_bytes(rows)
    manifest = {
        "schema": HELLASWAG_BUNDLE_SCHEMA,
        "dataset_name": HELLASWAG_DATASET_NAME,
        "split": HELLASWAG_SPLIT,
        "protocol": HELLASWAG_PROTOCOL,
        "requested_source_revision": requested_source_revision,
        "requested_source_revision_kind": "unverified_remote_selector",
        "source_fingerprint": fingerprint,
        "datasets_version": datasets_version,
        "tokenizer": tokenizer,
        "validation_count": len(rows),
        "files": {HELLASWAG_ROWS_FILENAME: _file_record(row_bytes)},
        "bundle_sha256": _bundle_sha256(row_bytes),
    }
    _publish_bundle(destination, row_bytes, manifest)
    verified = validate_hellaswag_bundle_directory(
        destination,
        tokenizer_identity=tokenizer,
        datasets_version=datasets_version,
        requested_source_revision=requested_source_revision,
        expected_count=expected_count,
    )
    return _bundle_result(
        destination, verified, expected_count=expected_count, reused=False
    )


def validate_hellaswag_bundle_directory(
    bundle_path: str | Path,
    *,
    tokenizer_identity: Mapping[str, object] | None = None,
    datasets_version: str | None = None,
    requested_source_revision: str | None = None,
    expected_count: int = HELLASWAG_VALIDATION_COUNT,
) -> dict[str, object]:
    root = Path(bundle_path)
    rows_path = root / HELLASWAG_ROWS_FILENAME
    manifest_path = root / HELLASWAG_MANIFEST_FILENAME
    if not root.is_dir():
        raise FileNotFoundError(f"HellaSwag bundle directory does not exist: {root}")
    missing = [path.name for path in (rows_path, manifest_path) if not path.is_file()]
    if missing:
        raise ValueError(
            f"HellaSwag bundle is incomplete; missing {', '.join(missing)}"
        )
    try:
        manifest = _json_object(
            json.loads(manifest_path.read_text(encoding="utf-8")), "HellaSwag manifest"
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"HellaSwag manifest is invalid JSON: {exc.msg}") from exc
    _validate_manifest(manifest, expected_count=expected_count)
    if tokenizer_identity is not None and manifest["tokenizer"] != _json_object(
        tokenizer_identity, "tokenizer_identity"
    ):
        raise ValueError(
            "HellaSwag manifest tokenizer identity does not match the request"
        )
    if (
        datasets_version is not None
        and manifest["datasets_version"] != datasets_version
    ):
        raise ValueError(
            "HellaSwag manifest datasets_version does not match the request"
        )
    if (
        requested_source_revision is not None
        and manifest["requested_source_revision"] != requested_source_revision
    ):
        raise ValueError(
            "HellaSwag manifest requested source revision does not match the request"
        )
    payload = rows_path.read_bytes()
    record = _json_object(
        manifest["files"][HELLASWAG_ROWS_FILENAME], "HellaSwag file record"
    )
    if len(payload) != record["bytes"]:
        raise ValueError(
            "HellaSwag validation.jsonl byte count does not match its manifest"
        )
    if hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError(
            "HellaSwag validation.jsonl sha256 does not match its manifest"
        )
    rows = _read_jsonl(rows_path)
    normalized = _validate_prepared_rows(rows, expected_count=expected_count)
    if payload != _jsonl_bytes(normalized):
        raise ValueError("HellaSwag validation rows are not canonical JSONL")
    if _bundle_sha256(payload) != manifest["bundle_sha256"]:
        raise ValueError("HellaSwag bundle_sha256 does not match local rows")
    return manifest


def load_hellaswag_bundle(
    bundle_path: str | Path, *, expected_count: int = HELLASWAG_VALIDATION_COUNT
) -> tuple[list[dict[str, object]], dict[str, object]]:
    root = Path(bundle_path)
    manifest = validate_hellaswag_bundle_directory(root, expected_count=expected_count)
    rows = _validate_prepared_rows(
        _read_jsonl(root / HELLASWAG_ROWS_FILENAME), expected_count=expected_count
    )
    return (rows, manifest)


def _validate_prepared_rows(
    rows: Iterable[Mapping[str, object]], *, expected_count: int
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        value = _json_object(row, f"HellaSwag prepared row {row_index}")
        if set(value) != {"id", "source_index", "context", "continuations", "label"}:
            raise ValueError(f"HellaSwag prepared row {row_index} fields are invalid")
        if value["id"] != f"validation:{row_index}":
            raise ValueError(f"HellaSwag prepared row {row_index} ID is not canonical")
        if not isinstance(value["context"], str) or not value["context"]:
            raise ValueError(f"HellaSwag prepared row {row_index} context is invalid")
        continuations = value["continuations"]
        if (
            not isinstance(continuations, list)
            or len(continuations) != 4
            or any(
                (
                    not isinstance(item, str) or not item.startswith(" ")
                    for item in continuations
                )
            )
        ):
            raise ValueError(
                f"HellaSwag prepared row {row_index} continuations are invalid"
            )
        if type(value["label"]) is not int or value["label"] not in range(4):
            raise ValueError(f"HellaSwag prepared row {row_index} label is invalid")
        normalized.append(value)
    if len(normalized) != expected_count:
        raise ValueError(
            f"HellaSwag validation split requires exactly {expected_count} rows; observed {len(normalized)}"
        )
    return normalized


def _validate_manifest(manifest: dict[str, object], *, expected_count: int) -> None:
    required = {
        "schema",
        "dataset_name",
        "split",
        "protocol",
        "requested_source_revision",
        "requested_source_revision_kind",
        "source_fingerprint",
        "datasets_version",
        "tokenizer",
        "validation_count",
        "files",
        "bundle_sha256",
    }
    if set(manifest) != required:
        raise ValueError("HellaSwag manifest fields do not match the bundle schema")
    if (
        manifest["schema"] != HELLASWAG_BUNDLE_SCHEMA
        or manifest["dataset_name"] != HELLASWAG_DATASET_NAME
        or manifest["split"] != HELLASWAG_SPLIT
        or (manifest["protocol"] != HELLASWAG_PROTOCOL)
    ):
        raise ValueError("HellaSwag manifest protocol identity is invalid")
    if manifest["requested_source_revision_kind"] != "unverified_remote_selector":
        raise ValueError("HellaSwag manifest revision kind is invalid")
    _validate_revision(manifest["requested_source_revision"])
    if manifest["source_fingerprint"] is not None and (
        not isinstance(manifest["source_fingerprint"], str)
    ):
        raise ValueError("HellaSwag manifest source fingerprint is invalid")
    if (
        not isinstance(manifest["datasets_version"], str)
        or not manifest["datasets_version"]
    ):
        raise ValueError("HellaSwag manifest datasets_version is invalid")
    if not isinstance(manifest["tokenizer"], Mapping) or not manifest["tokenizer"]:
        raise ValueError("HellaSwag manifest tokenizer identity is invalid")
    if manifest["validation_count"] != expected_count:
        raise ValueError("HellaSwag manifest validation count is invalid")
    files = _json_object(manifest["files"], "HellaSwag manifest files")
    if set(files) != {HELLASWAG_ROWS_FILENAME}:
        raise ValueError("HellaSwag manifest files are invalid")
    record = _json_object(files[HELLASWAG_ROWS_FILENAME], "HellaSwag file record")
    if set(record) != {"bytes", "sha256"} or type(record["bytes"]) is not int:
        raise ValueError("HellaSwag validation file record is invalid")
    if not _is_sha256(record["sha256"]) or not _is_sha256(manifest["bundle_sha256"]):
        raise ValueError("HellaSwag manifest sha256 fields are invalid")


def _publish_bundle(
    destination: Path, rows: bytes, manifest: Mapping[str, object]
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    published = False
    try:
        (temporary / HELLASWAG_ROWS_FILENAME).write_bytes(rows)
        (temporary / HELLASWAG_MANIFEST_FILENAME).write_bytes(
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
                f"HellaSwag {path.name} line {line_number} is invalid JSON"
            ) from exc
        rows.append(_json_object(value, f"HellaSwag {path.name} line {line_number}"))
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
    digest.update(HELLASWAG_ROWS_FILENAME.encode("ascii"))
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
        "validation_count": expected_count,
        "reused": reused,
    }


def _datasets_module() -> object:
    try:
        return importlib.import_module("datasets")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "HellaSwag preparation requires the optional data dependency; install qwen-lora-experiment[data]"
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
