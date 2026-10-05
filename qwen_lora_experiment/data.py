"""CPU-only data preparation primitives for the reproducible pilot assets.

The functions in this module deliberately accept dataset rows and tokenizers as
arguments.  Importing the module therefore neither imports Transformers or
Datasets nor accesses the network.  The two explicitly named download helpers
perform their optional import only when a data-preparation caller requests it.
"""

from __future__ import annotations
from collections.abc import Iterable, Mapping, Sequence
import importlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import numpy as np
from .paths import canonical_json, sha256_bytes, sha256_file

WIKITEXT_DATASET_NAME = "Salesforce/wikitext"
WIKITEXT_DATASET_CONFIG = "wikitext-103-raw-v1"
WIKITEXT_SPLITS = ("train", "validation", "test")
PIQA_DATASET_NAME = "lighteval/piqa"
PIQA_VALIDATION_SPLIT = "validation"
PIQA_FILENAME = "piqa_validation.jsonl"
SUPPORTED_SEQUENCE_LENGTHS = (2048, 4096)
PIQA_FIELDS = ("id", "goal", "sol1", "sol2", "label")
NO_SPECIAL_TOKEN_POLICY = {
    "add_special_tokens": False,
    "extra_bos": False,
    "extra_eos": False,
    "chat_template": False,
}


def _local_tokenizer_file(model_path: str | Path, filename: str) -> Path:
    candidate = Path(model_path) / filename
    if not candidate.is_file():
        raise FileNotFoundError(
            f"local tokenizer is missing required {filename}: {candidate}"
        )
    return candidate


def require_supported_sequence_length(sequence_length: int) -> int:
    """Return one supported asset sequence length or raise a clear error."""
    if (
        type(sequence_length) is not int
        or sequence_length not in SUPPORTED_SEQUENCE_LENGTHS
    ):
        raise ValueError("sequence_length must be 2048 or 4096")
    return sequence_length


def wikitext_asset_path(
    data_root: str | Path, split: str, sequence_length: int
) -> Path:
    """Return the canonical, length-specific local WikiText asset path."""
    if split not in WIKITEXT_SPLITS:
        raise ValueError(f"split must be one of {WIKITEXT_SPLITS}; got {split!r}")
    sequence_length = require_supported_sequence_length(sequence_length)
    return Path(data_root) / f"wikitext_{split}_{sequence_length}.npy"


def piqa_asset_path(data_root: str | Path) -> Path:
    """Return the canonical local PIQA-validation JSONL asset path."""
    return Path(data_root) / PIQA_FILENAME


def join_wikitext_rows(rows: Iterable[Mapping[str, object]]) -> str:
    """Join WikiText ``text`` fields in source order using exactly ``"\\n\\n"``."""
    texts: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"WikiText row {index} must be a mapping")
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError(f"WikiText row {index}.text must be a string")
        texts.append(text)
    return "\n\n".join(texts)


def tokenize_wikitext_rows(
    rows: Iterable[Mapping[str, object]], tokenizer: object
) -> list[int]:
    """Tokenize the canonical joined text without adding BOS/EOS/special tokens."""
    if not callable(tokenizer):
        raise TypeError("tokenizer must be callable")
    encoded = tokenizer(join_wikitext_rows(rows), add_special_tokens=False)
    if not isinstance(encoded, Mapping):
        raise TypeError("tokenizer result must be a mapping containing input_ids")
    input_ids = encoded.get("input_ids")
    if not isinstance(input_ids, Sequence) or isinstance(input_ids, (str, bytes)):
        raise ValueError("tokenizer result input_ids must be a sequence")
    token_ids: list[int] = []
    for index, token_id in enumerate(input_ids):
        if type(token_id) is not int:
            raise ValueError(f"tokenizer input_ids[{index}] must be an integer")
        token_ids.append(token_id)
    return token_ids


def pack_token_ids(token_ids: Iterable[int], sequence_length: int) -> np.ndarray:
    """Pack full token blocks as ``int32`` and drop the incomplete final tail.

    ``sequence_length`` is intentionally generic here, which keeps this pure
    packing primitive easy to unit-test.  Persisted pilot assets are restricted
    to 2048 or 4096 by :func:`wikitext_asset_path` and configuration validation.
    """
    if type(sequence_length) is not int or sequence_length <= 0:
        raise ValueError("sequence_length must be a positive integer")
    values = np.fromiter(_iter_int32_token_ids(token_ids), dtype=np.int32)
    complete_token_count = values.size - values.size % sequence_length
    if complete_token_count == 0:
        return np.empty((0, sequence_length), dtype=np.int32)
    return values[:complete_token_count].reshape(-1, sequence_length)


def write_packed_token_blocks(
    token_ids: Iterable[int],
    path: str | Path,
    *,
    sequence_length: int,
    chunk_blocks: int = 1024,
) -> Path:
    """Stream one token iterable into a new immutable packed NPY asset.

    The input is read exactly once.  Only one ``sequence_length`` token block
    and a bounded copying chunk are held in Python/NumPy memory; the temporary
    raw stream is staged on the target filesystem until the final NPY shape is
    known.  Existing assets are byte-checked and never replaced.
    """
    sequence_length = require_supported_sequence_length(sequence_length)
    if type(chunk_blocks) is not int or chunk_blocks <= 0:
        raise ValueError("chunk_blocks must be a positive integer")
    destination = Path(path)
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"WikiText asset parent directory does not exist: {destination.parent}"
        )
    raw_descriptor, raw_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tokens.tmp", dir=destination.parent
    )
    raw_path = Path(raw_name)
    npy_descriptor, npy_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".npy.tmp", dir=destination.parent
    )
    os.close(npy_descriptor)
    npy_path = Path(npy_name)
    try:
        block = np.empty(sequence_length, dtype=np.int32)
        position = 0
        block_count = 0
        with os.fdopen(raw_descriptor, "wb") as raw_file:
            for token_id in _iter_int32_token_ids(token_ids):
                block[position] = token_id
                position += 1
                if position == sequence_length:
                    raw_file.write(block.tobytes(order="C"))
                    block_count += 1
                    position = 0
            raw_file.flush()
            os.fsync(raw_file.fileno())
        mapped = np.lib.format.open_memmap(
            npy_path, mode="w+", dtype=np.int32, shape=(block_count, sequence_length)
        )
        try:
            copied_blocks = 0
            bytes_per_chunk = (
                chunk_blocks * sequence_length * np.dtype(np.int32).itemsize
            )
            with raw_path.open("rb") as raw_file:
                while payload := raw_file.read(bytes_per_chunk):
                    values = np.frombuffer(payload, dtype=np.int32)
                    if values.size % sequence_length:
                        raise RuntimeError(
                            "streamed token staging file is not block-aligned"
                        )
                    next_block_count = values.size // sequence_length
                    mapped[copied_blocks : copied_blocks + next_block_count] = (
                        values.reshape(next_block_count, sequence_length)
                    )
                    copied_blocks += next_block_count
            if copied_blocks != block_count:
                raise RuntimeError(
                    "streamed token staging block count changed unexpectedly"
                )
            mapped.flush()
        finally:
            del mapped
        _publish_npy_temp_no_clobber(npy_path, destination)
        return destination
    finally:
        raw_path.unlink(missing_ok=True)
        npy_path.unlink(missing_ok=True)


def deterministic_permutation(block_count: int, *, seed: int = 42) -> np.ndarray:
    """Create the one deterministic train-block permutation used by the pilot."""
    if type(block_count) is not int or block_count < 0:
        raise ValueError("block_count must be a non-negative integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    return (
        np.random.default_rng(seed)
        .permutation(block_count)
        .astype(np.int64, copy=False)
    )


def permutation_sha256(permutation: Iterable[int] | np.ndarray) -> str:
    """Hash a permutation's canonical little-endian int64 binary representation."""
    values = np.asarray(
        list(permutation) if not isinstance(permutation, np.ndarray) else permutation
    )
    if values.ndim != 1:
        raise ValueError("permutation must be one-dimensional")
    if values.dtype.kind not in "iu":
        raise ValueError("permutation must contain integers")
    canonical = values.astype("<i8", copy=False)
    return sha256_bytes(canonical.tobytes(order="C"))


def tokenizer_identity(
    tokenizer: object,
    *,
    vocab_file: str | Path | None = None,
    config_file: str | Path | None = None,
) -> dict[str, object]:
    """Collect stable, JSON-safe tokenizer identity without importing Transformers."""
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    if not isinstance(init_kwargs, Mapping):
        init_kwargs = {}
    revision = init_kwargs.get("revision", getattr(tokenizer, "revision", None))
    special_token_ids = {
        name: getattr(tokenizer, name, None)
        for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
    }
    for name, value in special_token_ids.items():
        if value is not None and type(value) is not int:
            raise ValueError(f"tokenizer {name} must be an integer or None")
    name_or_path = getattr(tokenizer, "name_or_path", None)
    if name_or_path is not None and (not isinstance(name_or_path, str)):
        raise ValueError("tokenizer name_or_path must be a string or None")
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None and type(vocab_size) is not int:
        raise ValueError("tokenizer vocab_size must be an integer or None")
    identity: dict[str, object] = {
        "class_name": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "name_or_path": name_or_path,
        "revision": (
            revision if isinstance(revision, str) or revision is None else str(revision)
        ),
        "vocab_size": vocab_size,
        "special_token_ids": special_token_ids,
    }
    resolved_vocab_file = (
        vocab_file
        or getattr(tokenizer, "vocab_file", None)
        or init_kwargs.get("vocab_file")
    )
    resolved_config_file = config_file or init_kwargs.get("tokenizer_config_file")
    if resolved_vocab_file is not None:
        identity["vocab_sha256"] = sha256_file(Path(resolved_vocab_file))
    if resolved_config_file is not None:
        identity["config_sha256"] = sha256_file(Path(resolved_config_file))
    return identity


def dataset_split_identity(
    dataset: object, *, revision: str | None = None
) -> dict[str, object]:
    """Collect the source fingerprint/revision supplied by an injected dataset."""
    fingerprint = getattr(dataset, "_fingerprint", None)
    if fingerprint is not None and (not isinstance(fingerprint, str)):
        fingerprint = str(fingerprint)
    return {"fingerprint": fingerprint, "revision": revision}


def validate_piqa_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Validate and normalize PIQA rows into the exact local JSONL schema.

    The upstream ``lighteval/piqa`` rows do not contain an identifier, so a
    deterministic row-based identifier is generated when ``id`` is absent.
    """
    normalized: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"PIQA row {index} must be a mapping")
        unknown = [key for key in row if key not in PIQA_FIELDS]
        if unknown:
            rendered = ", ".join((repr(key) for key in unknown))
            raise ValueError(f"PIQA row {index} has unknown field(s): {rendered}")
        missing = [field for field in PIQA_FIELDS if field != "id" and field not in row]
        if missing:
            raise ValueError(
                f"PIQA row {index} is missing required field(s): {', '.join(missing)}"
            )
        identifier = row.get("id", f"piqa-{index:05d}")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"PIQA row {index}.id must be a non-empty string")
        result: dict[str, object] = {"id": identifier}
        for field in ("goal", "sol1", "sol2"):
            value = row[field]
            if not isinstance(value, str):
                raise ValueError(f"PIQA row {index}.{field} must be a string")
            result[field] = value
        label = row["label"]
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f"PIQA row {index}.label must be 0 or 1")
        result["label"] = label
        normalized.append(result)
    return normalized


def serialize_piqa_rows(rows: Iterable[Mapping[str, object]], path: str | Path) -> Path:
    """Write PIQA rows once, or verify an identical existing immutable asset."""
    destination = Path(path)
    normalized = validate_piqa_rows(rows)
    encoded = "".join((canonical_json(row) + "\n" for row in normalized)).encode(
        "utf-8"
    )
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"PIQA asset parent directory does not exist: {destination.parent}"
        )
    _publish_bytes_no_clobber(destination, encoded, asset_name="PIQA asset")
    return destination


def load_piqa_rows(path: str | Path) -> list[dict[str, object]]:
    """Read and validate the canonical PIQA JSONL asset without downloads."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"PIQA asset does not exist: {source}")
    rows: list[dict[str, object]] = []
    with source.open(encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            if not line.endswith("\n"):
                raise ValueError(
                    f"PIQA JSONL line {line_number} must end with a newline"
                )
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"PIQA JSONL line {line_number} is invalid: {exc.msg}"
                ) from exc
            if not isinstance(decoded, Mapping):
                raise ValueError(f"PIQA JSONL line {line_number} must be an object")
            rows.extend(validate_piqa_rows([decoded]))
    return rows


def build_piqa_prompt_rows(
    rows: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Build the fixed PIQA prompt/continuation records used by evaluation."""
    prompts: list[dict[str, object]] = []
    for row in validate_piqa_rows(rows):
        goal = str(row["goal"]).strip()
        solution_one = str(row["sol1"]).strip()
        solution_two = str(row["sol2"]).strip()
        prompts.append(
            {
                "id": row["id"],
                "label": row["label"],
                "prompt": f"Question: {goal}\nAnswer:",
                "continuations": (f" {solution_one}", f" {solution_two}"),
            }
        )
    return prompts


def download_wikitext_splits(*, revision: str | None = None) -> dict[str, object]:
    """Download the three declared WikiText splits on explicit caller request only."""
    datasets = _import_optional_datasets()
    return {
        split: datasets.load_dataset(
            WIKITEXT_DATASET_NAME,
            WIKITEXT_DATASET_CONFIG,
            split=split,
            revision=revision,
        )
        for split in WIKITEXT_SPLITS
    }


def download_piqa_validation(*, revision: str | None = None) -> object:
    """Download PIQA validation on explicit caller request only."""
    datasets = _import_optional_datasets()
    return datasets.load_dataset(
        PIQA_DATASET_NAME, split=PIQA_VALIDATION_SPLIT, revision=revision
    )


def _import_optional_datasets() -> Any:
    try:
        return importlib.import_module("datasets")
    except ImportError as exc:
        raise RuntimeError(
            "dataset download requires the optional 'datasets' package"
        ) from exc


def _iter_int32_token_ids(token_ids: Iterable[int]) -> Iterable[int]:
    for index, token_id in enumerate(token_ids):
        if type(token_id) is not int:
            raise ValueError(f"token_ids[{index}] must be an integer")
        if not np.iinfo(np.int32).min <= token_id <= np.iinfo(np.int32).max:
            raise ValueError(f"token_ids[{index}] is outside int32 range")
        yield token_id


def _publish_bytes_no_clobber(
    destination: Path, content: bytes, *, asset_name: str
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        try:
            os.link(temporary_path, destination)
        except FileExistsError:
            if not destination.is_file():
                raise ValueError(f"{asset_name} is not a regular file: {destination}")
            if destination.read_bytes() != content:
                raise ValueError(
                    f"existing {asset_name} does not match requested content: {destination}"
                )
        else:
            _fsync_directory(destination.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    temporary_path.unlink(missing_ok=True)


def _publish_npy_temp_no_clobber(temporary_path: Path, destination: Path) -> None:
    try:
        os.link(temporary_path, destination)
    except FileExistsError:
        if not destination.is_file():
            raise ValueError(f"WikiText asset is not a regular file: {destination}")
        if sha256_file(destination) != sha256_file(temporary_path):
            raise ValueError(
                f"existing WikiText asset does not match requested content: {destination}"
            )
    else:
        _fsync_directory(destination.parent)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
