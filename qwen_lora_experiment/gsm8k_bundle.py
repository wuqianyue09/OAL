"""Create-only disk bundle for the dependency-light GSM8K protocol."""

from __future__ import annotations
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from .gsm8k import (
    GSM8K_BUNDLE_DIRECTORY,
    GSM8K_BUNDLE_SCHEMA,
    GSM8K_CONFIG_NAME,
    GSM8K_DATASET_NAME,
    GSM8K_FEWSHOT_COUNT,
    GSM8K_FEWSHOT_FILENAME,
    GSM8K_FEWSHOT_SEED,
    GSM8K_MANIFEST_FILENAME,
    GSM8K_PROTOCOL,
    GSM8K_TEST_COUNT,
    GSM8K_TEST_FILENAME,
    GSM8K_TRAIN_COUNT,
    Gsm8kSourceLoader,
    _json_object,
    _question_answer,
    _validate_prepared_test_rows,
    build_fewshot_assignments,
    normalize_gsm8k_test_rows,
    normalize_gsm8k_train_rows,
)


def gsm8k_bundle_path(data_root: str | Path) -> Path:
    return Path(data_root) / "gsm8k" / GSM8K_BUNDLE_DIRECTORY


def load_gsm8k_source(
    split: str, requested_source_revision: str | None
) -> Iterable[Mapping[str, object]]:
    if split not in {"train", "test"}:
        raise ValueError("GSM8K source split must be train or test")
    datasets = _datasets_module()
    loader = getattr(datasets, "load_dataset", None)
    if not callable(loader):
        raise RuntimeError("installed datasets package does not provide load_dataset")
    kwargs: dict[str, object] = {}
    if requested_source_revision is not None:
        kwargs["revision"] = requested_source_revision
    return loader(GSM8K_DATASET_NAME, GSM8K_CONFIG_NAME, split=split, **kwargs)


def installed_datasets_version() -> str:
    version = getattr(_datasets_module(), "__version__", None)
    if not isinstance(version, str) or not version:
        raise RuntimeError("installed datasets package does not expose a version")
    return version


def prepare_gsm8k_bundle(
    data_root: str | Path,
    *,
    source_loader: Gsm8kSourceLoader | None,
    tokenizer_identity: Mapping[str, object],
    datasets_version: str | None = None,
    requested_source_revision: str | None = None,
    verify_only: bool = False,
    expected_train_count: int = GSM8K_TRAIN_COUNT,
    expected_test_count: int = GSM8K_TEST_COUNT,
) -> dict[str, object]:
    _validate_revision(requested_source_revision)
    tokenizer = _json_object(tokenizer_identity, "tokenizer_identity")
    if not tokenizer:
        raise ValueError("tokenizer_identity must not be empty")
    destination = gsm8k_bundle_path(data_root)
    if destination.exists():
        manifest = validate_gsm8k_bundle_directory(
            destination,
            tokenizer_identity=tokenizer,
            datasets_version=datasets_version,
            requested_source_revision=requested_source_revision,
            expected_train_count=expected_train_count,
            expected_test_count=expected_test_count,
        )
        return _bundle_result(destination, manifest, expected_test_count, reused=True)
    if verify_only:
        raise FileNotFoundError(
            f"GSM8K bundle does not exist for verify-only: {destination}"
        )
    if source_loader is None:
        raise ValueError("source_loader is required to create a missing GSM8K bundle")
    if datasets_version is None:
        datasets_version = installed_datasets_version()
    if not isinstance(datasets_version, str) or not datasets_version:
        raise ValueError("datasets_version must be a non-empty string")
    train_source = source_loader("train", requested_source_revision)
    test_source = source_loader("test", requested_source_revision)
    fingerprints = {
        "train": _source_fingerprint(train_source),
        "test": _source_fingerprint(test_source),
    }
    train_rows = normalize_gsm8k_train_rows(
        train_source, expected_count=expected_train_count
    )
    test_rows = normalize_gsm8k_test_rows(
        test_source, expected_count=expected_test_count
    )
    assignments = build_fewshot_assignments(
        train_rows, test_rows, seed=GSM8K_FEWSHOT_SEED
    )
    fewshot_bytes = _jsonl_bytes(assignments)
    test_bytes = _jsonl_bytes(test_rows)
    manifest = {
        "schema": GSM8K_BUNDLE_SCHEMA,
        "dataset_name": GSM8K_DATASET_NAME,
        "config_name": GSM8K_CONFIG_NAME,
        "protocol": GSM8K_PROTOCOL,
        "requested_source_revision": requested_source_revision,
        "requested_source_revision_kind": "unverified_remote_selector",
        "source_fingerprints": fingerprints,
        "datasets_version": datasets_version,
        "tokenizer": tokenizer,
        "train_count": len(train_rows),
        "test_count": len(test_rows),
        "fewshot_count": GSM8K_FEWSHOT_COUNT,
        "fewshot_seed": GSM8K_FEWSHOT_SEED,
        "fewshot_sampler": "python_random_sample_sequential_per_test",
        "test_order": "canonical_source_order",
        "files": {
            GSM8K_FEWSHOT_FILENAME: _file_record(fewshot_bytes),
            GSM8K_TEST_FILENAME: _file_record(test_bytes),
        },
        "bundle_sha256": _bundle_sha256(fewshot_bytes, test_bytes),
    }
    _publish_bundle(destination, fewshot_bytes, test_bytes, manifest)
    verified = validate_gsm8k_bundle_directory(
        destination,
        tokenizer_identity=tokenizer,
        datasets_version=datasets_version,
        requested_source_revision=requested_source_revision,
        expected_train_count=expected_train_count,
        expected_test_count=expected_test_count,
    )
    return _bundle_result(destination, verified, expected_test_count, reused=False)


def validate_gsm8k_bundle_directory(
    bundle_path: str | Path,
    *,
    tokenizer_identity: Mapping[str, object] | None = None,
    datasets_version: str | None = None,
    requested_source_revision: str | None = None,
    expected_train_count: int = GSM8K_TRAIN_COUNT,
    expected_test_count: int = GSM8K_TEST_COUNT,
) -> dict[str, object]:
    root = Path(bundle_path)
    paths = {
        GSM8K_FEWSHOT_FILENAME: root / GSM8K_FEWSHOT_FILENAME,
        GSM8K_TEST_FILENAME: root / GSM8K_TEST_FILENAME,
        GSM8K_MANIFEST_FILENAME: root / GSM8K_MANIFEST_FILENAME,
    }
    if not root.is_dir():
        raise FileNotFoundError(f"GSM8K bundle directory does not exist: {root}")
    missing = [name for (name, path) in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"GSM8K bundle is incomplete; missing {', '.join(missing)}")
    try:
        manifest = _json_object(
            json.loads(paths[GSM8K_MANIFEST_FILENAME].read_text(encoding="utf-8")),
            "GSM8K manifest",
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"GSM8K manifest is invalid JSON: {exc.msg}") from exc
    _validate_manifest(
        manifest,
        expected_train_count=expected_train_count,
        expected_test_count=expected_test_count,
    )
    if tokenizer_identity is not None and manifest["tokenizer"] != _json_object(
        tokenizer_identity, "tokenizer_identity"
    ):
        raise ValueError("GSM8K manifest tokenizer identity does not match the request")
    if (
        datasets_version is not None
        and manifest["datasets_version"] != datasets_version
    ):
        raise ValueError("GSM8K manifest datasets_version does not match the request")
    if (
        requested_source_revision is not None
        and manifest["requested_source_revision"] != requested_source_revision
    ):
        raise ValueError(
            "GSM8K manifest requested source revision does not match the request"
        )
    payloads = {
        name: paths[name].read_bytes()
        for name in (GSM8K_FEWSHOT_FILENAME, GSM8K_TEST_FILENAME)
    }
    files = _json_object(manifest["files"], "GSM8K manifest files")
    for name, payload in payloads.items():
        record = _json_object(files[name], f"GSM8K {name} file record")
        if len(payload) != record["bytes"]:
            raise ValueError(f"GSM8K {name} byte count does not match its manifest")
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError(f"GSM8K {name} sha256 does not match its manifest")
    test_rows = _validate_prepared_test_rows(_read_jsonl(paths[GSM8K_TEST_FILENAME]))
    if len(test_rows) != expected_test_count:
        raise ValueError("GSM8K test row count does not match the request")
    assignments = _validate_assignments(
        _read_jsonl(paths[GSM8K_FEWSHOT_FILENAME]),
        test_rows,
        expected_train_count=expected_train_count,
    )
    if len(assignments) != expected_test_count:
        raise ValueError("GSM8K few-shot assignment count does not match the request")
    if payloads[GSM8K_TEST_FILENAME] != _jsonl_bytes(test_rows) or payloads[
        GSM8K_FEWSHOT_FILENAME
    ] != _jsonl_bytes(assignments):
        raise ValueError("GSM8K rows are not canonical JSONL")
    if (
        _bundle_sha256(payloads[GSM8K_FEWSHOT_FILENAME], payloads[GSM8K_TEST_FILENAME])
        != manifest["bundle_sha256"]
    ):
        raise ValueError("GSM8K bundle_sha256 does not match local rows")
    return manifest


def load_gsm8k_bundle(
    bundle_path: str | Path,
    *,
    expected_train_count: int = GSM8K_TRAIN_COUNT,
    expected_test_count: int = GSM8K_TEST_COUNT,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    root = Path(bundle_path)
    manifest = validate_gsm8k_bundle_directory(
        root,
        expected_train_count=expected_train_count,
        expected_test_count=expected_test_count,
    )
    test_rows = _validate_prepared_test_rows(_read_jsonl(root / GSM8K_TEST_FILENAME))
    assignments = _validate_assignments(
        _read_jsonl(root / GSM8K_FEWSHOT_FILENAME),
        test_rows,
        expected_train_count=expected_train_count,
    )
    return (assignments, test_rows, manifest)


def _validate_assignments(
    assignments: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
    *,
    expected_train_count: int,
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    if len(assignments) != len(test_rows):
        raise ValueError("GSM8K assignments must cover every test row")
    for index, (assignment, test_row) in enumerate(
        zip(assignments, test_rows, strict=True)
    ):
        value = _json_object(assignment, f"GSM8K assignment {index}")
        required = {"id", "test_index", "demonstration_indices", "demonstrations"}
        if (
            set(value) != required
            or value["id"] != test_row["id"]
            or value["test_index"] != index
        ):
            raise ValueError(f"GSM8K assignment {index} identity is invalid")
        indices = value["demonstration_indices"]
        demonstrations = value["demonstrations"]
        if (
            not isinstance(indices, list)
            or len(indices) != GSM8K_FEWSHOT_COUNT
            or len(set(indices)) != GSM8K_FEWSHOT_COUNT
            or any(
                (
                    type(item) is not int or item not in range(expected_train_count)
                    for item in indices
                )
            )
        ):
            raise ValueError(f"GSM8K assignment {index} indices are invalid")
        if (
            not isinstance(demonstrations, list)
            or len(demonstrations) != GSM8K_FEWSHOT_COUNT
        ):
            raise ValueError(f"GSM8K assignment {index} demonstrations are invalid")
        for demo_index, (train_index, demonstration) in enumerate(
            zip(indices, demonstrations, strict=True)
        ):
            demo = _json_object(
                demonstration, f"GSM8K assignment {index} demo {demo_index}"
            )
            if (
                set(demo) != {"id", "question", "answer"}
                or demo["id"] != f"train:{train_index}"
            ):
                raise ValueError(
                    f"GSM8K assignment {index} demo {demo_index} is invalid"
                )
            _question_answer(demo, f"assignment {index} demo {demo_index}")
        normalized.append(value)
    return normalized


def _validate_manifest(
    manifest: dict[str, object], *, expected_train_count: int, expected_test_count: int
) -> None:
    required = {
        "schema",
        "dataset_name",
        "config_name",
        "protocol",
        "requested_source_revision",
        "requested_source_revision_kind",
        "source_fingerprints",
        "datasets_version",
        "tokenizer",
        "train_count",
        "test_count",
        "fewshot_count",
        "fewshot_seed",
        "fewshot_sampler",
        "test_order",
        "files",
        "bundle_sha256",
    }
    if set(manifest) != required:
        raise ValueError("GSM8K manifest fields do not match the bundle schema")
    if (
        manifest["schema"] != GSM8K_BUNDLE_SCHEMA
        or manifest["dataset_name"] != GSM8K_DATASET_NAME
        or manifest["config_name"] != GSM8K_CONFIG_NAME
        or (manifest["protocol"] != GSM8K_PROTOCOL)
    ):
        raise ValueError("GSM8K manifest protocol identity is invalid")
    if manifest["requested_source_revision_kind"] != "unverified_remote_selector":
        raise ValueError("GSM8K manifest revision kind is invalid")
    _validate_revision(manifest["requested_source_revision"])
    fingerprints = _json_object(manifest["source_fingerprints"], "source_fingerprints")
    if set(fingerprints) != {"train", "test"} or any(
        (
            value is not None and (not isinstance(value, str))
            for value in fingerprints.values()
        )
    ):
        raise ValueError("GSM8K manifest source fingerprints are invalid")
    if (
        not isinstance(manifest["datasets_version"], str)
        or not manifest["datasets_version"]
    ):
        raise ValueError("GSM8K manifest datasets_version is invalid")
    if not isinstance(manifest["tokenizer"], Mapping) or not manifest["tokenizer"]:
        raise ValueError("GSM8K manifest tokenizer identity is invalid")
    expected = {
        "train_count": expected_train_count,
        "test_count": expected_test_count,
        "fewshot_count": GSM8K_FEWSHOT_COUNT,
        "fewshot_seed": GSM8K_FEWSHOT_SEED,
        "fewshot_sampler": "python_random_sample_sequential_per_test",
        "test_order": "canonical_source_order",
    }
    for field, value in expected.items():
        if manifest[field] != value:
            raise ValueError(f"GSM8K manifest {field} is invalid")
    files = _json_object(manifest["files"], "GSM8K manifest files")
    if set(files) != {GSM8K_FEWSHOT_FILENAME, GSM8K_TEST_FILENAME}:
        raise ValueError("GSM8K manifest files are invalid")
    for name in files:
        record = _json_object(files[name], f"GSM8K {name} file record")
        if set(record) != {"bytes", "sha256"} or type(record["bytes"]) is not int:
            raise ValueError(f"GSM8K {name} file record is invalid")
        if not _is_sha256(record["sha256"]):
            raise ValueError(f"GSM8K {name} sha256 is invalid")
    if not _is_sha256(manifest["bundle_sha256"]):
        raise ValueError("GSM8K manifest bundle sha256 is invalid")


def _publish_bundle(
    destination: Path,
    fewshot_bytes: bytes,
    test_bytes: bytes,
    manifest: Mapping[str, object],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    published = False
    try:
        (temporary / GSM8K_FEWSHOT_FILENAME).write_bytes(fewshot_bytes)
        (temporary / GSM8K_TEST_FILENAME).write_bytes(test_bytes)
        (temporary / GSM8K_MANIFEST_FILENAME).write_bytes(
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
                f"GSM8K {path.name} line {line_number} is invalid JSON"
            ) from exc
        rows.append(_json_object(value, f"GSM8K {path.name} line {line_number}"))
    return rows


def _jsonl_bytes(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join((_canonical_json_bytes(row) + b"\n" for row in rows))


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _file_record(payload: bytes) -> dict[str, object]:
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _bundle_sha256(fewshot_bytes: bytes, test_bytes: bytes) -> str:
    digest = hashlib.sha256()
    for name, payload in (
        (GSM8K_FEWSHOT_FILENAME, fewshot_bytes),
        (GSM8K_TEST_FILENAME, test_bytes),
    ):
        digest.update(name.encode("ascii"))
        digest.update(b"\x00")
        digest.update(payload)
    return digest.hexdigest()


def _bundle_result(
    destination: Path, manifest: Mapping[str, object], test_count: int, *, reused: bool
) -> dict[str, object]:
    return {
        "bundle_path": str(destination),
        "bundle_sha256": manifest["bundle_sha256"],
        "test_count": test_count,
        "reused": reused,
    }


def _source_fingerprint(source: object) -> str | None:
    value = getattr(source, "_fingerprint", None)
    return value if isinstance(value, str) else None


def _datasets_module() -> object:
    try:
        return importlib.import_module("datasets")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "GSM8K preparation requires the optional data dependency; install qwen-lora-experiment[data]"
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
