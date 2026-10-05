"""Prepared training data manifests, token blocks, and seeded selections."""

from __future__ import annotations
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from .config import DEFAULT_SEQUENCE_LENGTH, PilotConfig
from .data import (
    NO_SPECIAL_TOKEN_POLICY,
    PIQA_DATASET_NAME,
    PIQA_FILENAME,
    PIQA_VALIDATION_SPLIT,
    WIKITEXT_DATASET_CONFIG,
    WIKITEXT_DATASET_NAME,
    WIKITEXT_SPLITS,
    deterministic_permutation,
    load_piqa_rows,
    permutation_sha256,
    piqa_asset_path,
    require_supported_sequence_length,
    wikitext_asset_path,
)
from .experiment_contract import FORMAL_MASTER_SEEDS
from .paths import canonical_json, sha256_file
from .protocol import resolve_seed_derivations
import hashlib
import json
import numpy as np
import os
import tempfile
from .asset_io import (
    _fsync_directory,
    _require_nonempty_string,
    _validate_json_object,
    _write_temporary_bytes,
)

MANIFEST_FILENAME = "manifest.json"
DATA_MANIFEST_SCHEMA_VERSION = 2
_RULES = {
    "text_joiner": "\n\n",
    "add_special_tokens": False,
    "extra_bos_eos": False,
    "tail": "drop_incomplete_block",
    "block_dtype": "int32",
}


def data_manifest_path(data_root: str | Path, sequence_length: int) -> Path:
    """Return the immutable manifest path for one supported sequence length.

    The default 2048 profile retains the documented ``manifest.json`` name.
    A common 4096 fallback stores its independent provenance beside it rather
    than overwriting 2048's block-selection and tokenizer evidence.
    """
    sequence_length = require_supported_sequence_length(sequence_length)
    filename = (
        MANIFEST_FILENAME
        if sequence_length == DEFAULT_SEQUENCE_LENGTH
        else f"manifest_{sequence_length}.json"
    )
    return Path(data_root) / filename


def training_permutation_from_manifest(
    manifest: Mapping[str, object], seed: int
) -> np.ndarray:
    """Read the frozen parent train order for one master seed without opening assets."""
    try:
        wikitext = manifest["wikitext"]
        train = wikitext["train"]
        schedules = train["permutations_by_master_seed"]
        selected = schedules[str(seed)]
        values = selected["permutation"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "training data manifest is missing the parent train permutation"
        ) from exc
    return np.asarray(values, dtype=np.int64)


def migrate_legacy_data_manifest(
    data_root: str | Path, *, sequence_length: int
) -> dict[str, object]:
    """Explicitly promote an intact schema-v1 manifest to schema v2.

    Schema v2 records the complete 17/42/73 formal permutation schedule;
    schema v1 recorded only its legacy seed-42 permutation.  This migration
    validates the old manifest and every referenced local asset before deriving
    the v2 schedule.  It never discards the old evidence: the original bytes
    are hard-linked as ``*.schema-v1.json`` before the v2 manifest is atomically
    installed at the canonical manifest path.

    It is intentionally an explicit maintenance operation, never an implicit
    fallback in a training or smoke execution path.
    """
    root = Path(data_root)
    path = data_manifest_path(root, sequence_length)
    if not path.is_file():
        raise FileNotFoundError(f"data manifest does not exist: {path}")
    try:
        original_bytes = path.read_bytes()
        raw = json.loads(original_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"data manifest JSON is invalid: {exc.msg}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("data manifest must be a JSON object")
    schema_version = raw.get("schema_version")
    if schema_version == DATA_MANIFEST_SCHEMA_VERSION:
        manifest = validate_data_manifest(path, sequence_length=sequence_length)
        return {
            "status": "already_current",
            "manifest_path": str(path),
            "manifest_sha256": sha256_file(path),
            "sequence_length": manifest["sequence_length"],
        }
    legacy = _validate_legacy_v1_manifest_structure(raw)
    asset_length = legacy["sequence_length"]
    assert type(asset_length) is int
    if asset_length != require_supported_sequence_length(sequence_length):
        raise ValueError(
            f"manifest sequence_length {asset_length} does not match requested {sequence_length}"
        )
    _validate_manifest_against_assets(legacy, root, include_piqa=True)
    upgraded = _upgrade_legacy_v1_manifest(legacy)
    encoded = (canonical_json(upgraded) + "\n").encode("utf-8")
    backup = path.with_name(f"{path.stem}.schema-v1{path.suffix}")
    if backup.exists():
        if not backup.is_file() or backup.read_bytes() != original_bytes:
            raise FileExistsError(
                f"legacy manifest backup already exists and does not match the source: {backup}"
            )
    else:
        try:
            os.link(path, backup)
        except FileExistsError:
            if not backup.is_file() or backup.read_bytes() != original_bytes:
                raise FileExistsError(
                    f"legacy manifest backup already exists and does not match the source: {backup}"
                )
        else:
            _fsync_directory(root)
    temporary_path = _write_temporary_bytes(path, encoded)
    try:
        if path.read_bytes() != original_bytes:
            raise RuntimeError("legacy manifest changed during migration")
        os.replace(temporary_path, path)
        _fsync_directory(root)
    finally:
        temporary_path.unlink(missing_ok=True)
    migrated = validate_data_manifest(path, sequence_length=sequence_length)
    return {
        "status": "migrated_schema_v1_to_v2",
        "legacy_manifest_path": str(backup),
        "legacy_manifest_sha256": sha256_file(backup),
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path),
        "sequence_length": migrated["sequence_length"],
    }


def selected_token_block_sha256(values: np.ndarray, block_ids: Sequence[int]) -> str:
    """Hash exactly the selected little-endian int32 blocks in row order.

    This is shared by formal asset creation and its live admission check, so
    block identifiers are recorded beside the hash rather than mixed into it.
    """
    if (
        not isinstance(values, np.ndarray)
        or values.ndim != 2
        or values.dtype != np.int32
        or (not block_ids)
        or any((type(block_id) is not int or block_id < 0 for block_id in block_ids))
        or (len(set(block_ids)) != len(block_ids))
        or (max(block_ids) >= values.shape[0])
    ):
        raise ValueError("selected token block IDs do not match an int32 block array")
    selected = np.asarray(values[list(block_ids)], dtype=np.dtype("<i4"))
    return hashlib.sha256(selected.tobytes(order="C")).hexdigest()


def build_data_manifest(
    *,
    sequence_length: int,
    tokenizer: Mapping[str, object],
    model_profile: str | None = None,
    model_source: str | Path | None = None,
    wikitext: Mapping[str, Mapping[str, object]],
    piqa_provenance: Mapping[str, object],
    data_root: str | Path,
    train_permutations_by_master_seed: (
        Mapping[str, Iterable[int] | np.ndarray] | None
    ) = None,
) -> dict[str, object]:
    """Build one self-checking manifest from already-created local assets.

    This function never writes data files.  It hashes and shape-checks the
    supplied assets first, allowing callers to persist a manifest only after
    the complete asset set exists.
    """
    sequence_length = require_supported_sequence_length(sequence_length)
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"data asset root does not exist: {root}")
    normalized_tokenizer = _validate_tokenizer_identity(tokenizer)
    split_metadata = _validate_wikitext_metadata(wikitext)
    normalized_piqa_provenance = _validate_piqa_provenance(piqa_provenance)
    files: dict[str, dict[str, object]] = {}
    split_records: dict[str, dict[str, object]] = {}
    train_block_count: int | None = None
    for split in WIKITEXT_SPLITS:
        path = wikitext_asset_path(root, split, sequence_length)
        blocks = _load_int32_blocks(path, sequence_length)
        block_count = int(blocks.shape[0])
        if split == "train":
            train_block_count = block_count
        filename = path.name
        files[filename] = _file_record(path)
        metadata = dict(split_metadata[split])
        metadata.update(
            {
                "dataset_name": WIKITEXT_DATASET_NAME,
                "dataset_config": WIKITEXT_DATASET_CONFIG,
                "block_count": block_count,
                "packed_token_count": block_count * sequence_length,
                "filename": filename,
            }
        )
        split_records[split] = metadata
    assert train_block_count is not None
    split_records["train"]["permutations_by_master_seed"] = (
        _build_train_permutations_by_master_seed(
            train_block_count, supplied=train_permutations_by_master_seed
        )
    )
    piqa_path = piqa_asset_path(root)
    piqa_rows = load_piqa_rows(piqa_path)
    files[PIQA_FILENAME] = _file_record(piqa_path)
    manifest = {
        "schema_version": DATA_MANIFEST_SCHEMA_VERSION,
        "kind": "qwen_lora_prepared_data",
        "sequence_length": sequence_length,
        "tokenizer": normalized_tokenizer,
        "rules": dict(_RULES),
        "wikitext": split_records,
        "piqa": {
            "dataset_name": PIQA_DATASET_NAME,
            "split": PIQA_VALIDATION_SPLIT,
            "fingerprint": normalized_piqa_provenance["fingerprint"],
            "revision": normalized_piqa_provenance["revision"],
            "filename": PIQA_FILENAME,
            "row_count": len(piqa_rows),
        },
        "files": dict(sorted(files.items())),
    }
    if model_profile is not None or model_source is not None:
        if model_profile is None or model_source is None:
            raise ValueError("model_profile and model_source must be provided together")
        from .backbones import geometry_for_profile

        geometry_for_profile(model_profile)
        manifest["model_profile"] = model_profile
        manifest["model_source"] = str(Path(model_source).resolve())
        manifest["special_token_policy"] = dict(NO_SPECIAL_TOKEN_POLICY)
    return _validate_manifest_structure(manifest)


def write_data_manifest(path: str | Path, manifest: Mapping[str, object]) -> Path:
    """Persist a new manifest, or validate an existing identical manifest.

    An existing manifest is intentionally never overwritten.  This catches an
    attempt to silently reuse a data root for different source/configuration
    facts, rather than replacing its provenance evidence.
    """
    destination = Path(path)
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"manifest parent directory does not exist: {destination.parent}"
        )
    proposed = _validate_manifest_structure(manifest)
    _validate_manifest_against_assets(proposed, destination.parent, include_piqa=True)
    encoded = (canonical_json(proposed) + "\n").encode("utf-8")
    temporary_path = _write_temporary_bytes(destination, encoded)
    try:
        try:
            os.link(temporary_path, destination)
        except FileExistsError:
            existing = validate_data_manifest(destination)
            if canonical_json(existing) != canonical_json(proposed):
                raise ValueError(
                    f"existing manifest does not match requested manifest: {destination}"
                )
        else:
            _fsync_directory(destination.parent)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def validate_data_manifest(
    path: str | Path,
    *,
    sequence_length: int | None = None,
    tokenizer: Mapping[str, object] | None = None,
    include_piqa: bool = True,
    model_profile: str | None = None,
    model_source: str | Path | None = None,
) -> dict[str, object]:
    """Validate manifest structure and the requested endpoint's local assets.

    NLL is a standalone first stage.  It must validate all WikiText identity
    evidence without parsing or even hashing the deferred PIQA JSONL; PIQA is
    validated in full only after its formal paired-NLL report gate succeeds.
    The default remains the complete validation used by data preparation and
    the authorized PIQA endpoint.
    """
    manifest_path = Path(path)
    if type(include_piqa) is not bool:
        raise TypeError("include_piqa must be a boolean")
    manifest = _read_data_manifest_structure(manifest_path)
    asset_length = manifest["sequence_length"]
    assert type(asset_length) is int
    if (
        sequence_length is not None
        and asset_length != require_supported_sequence_length(sequence_length)
    ):
        raise ValueError(
            f"manifest sequence_length {asset_length} does not match requested {sequence_length}"
        )
    if tokenizer is not None and canonical_json(
        manifest["tokenizer"]
    ) != canonical_json(dict(tokenizer)):
        raise ValueError(
            "manifest tokenizer identity does not match requested tokenizer"
        )
    _validate_manifest_model_binding(
        manifest, model_profile=model_profile, model_source=model_source
    )
    _validate_manifest_against_assets(
        manifest, manifest_path.parent, include_piqa=include_piqa
    )
    return dict(manifest)


def _validate_manifest_against_assets(
    manifest: Mapping[str, object], root: Path, *, include_piqa: bool
) -> None:
    """Check one already-structured manifest against the local immutable files."""
    asset_length = manifest["sequence_length"]
    assert type(asset_length) is int
    files = manifest["files"]
    assert isinstance(files, Mapping)
    for filename, record in files.items():
        assert isinstance(filename, str)
        assert isinstance(record, Mapping)
        if filename == PIQA_FILENAME and (not include_piqa):
            continue
        candidate = root / filename
        actual_hash = sha256_file(candidate)
        expected_hash = record["sha256"]
        assert isinstance(expected_hash, str)
        if actual_hash != expected_hash:
            raise ValueError(f"SHA-256 mismatch for asset {candidate}")
        if candidate.stat().st_size != record["bytes"]:
            raise ValueError(f"byte count mismatch for asset {candidate}")
    wikitext = manifest["wikitext"]
    assert isinstance(wikitext, Mapping)
    for split in WIKITEXT_SPLITS:
        split_record = wikitext[split]
        assert isinstance(split_record, Mapping)
        filename = split_record["filename"]
        assert isinstance(filename, str)
        expected_filename = wikitext_asset_path(root, split, asset_length).name
        if filename != expected_filename:
            raise ValueError(f"manifest {split} filename must be {expected_filename}")
        blocks = _load_int32_blocks(root / filename, asset_length)
        if blocks.shape[0] != split_record["block_count"]:
            raise ValueError(f"manifest {split} block_count does not match asset shape")
        if split_record["packed_token_count"] != int(blocks.shape[0]) * asset_length:
            raise ValueError(
                f"manifest {split} packed_token_count does not match block packing"
            )
    if include_piqa:
        piqa = manifest["piqa"]
        assert isinstance(piqa, Mapping)
        rows = load_piqa_rows(root / PIQA_FILENAME)
        if piqa["row_count"] != len(rows):
            raise ValueError("manifest PIQA row_count does not match JSONL asset")


def write_wikitext_blocks(
    path: str | Path, blocks: np.ndarray, *, sequence_length: int
) -> Path:
    """Write a new immutable int32 block asset or validate identical bytes."""
    sequence_length = require_supported_sequence_length(sequence_length)
    destination = Path(path)
    array = np.asarray(blocks)
    if array.dtype != np.int32 or array.ndim != 2 or array.shape[1] != sequence_length:
        raise ValueError(
            f"WikiText blocks must be int32 with shape [num_blocks, {sequence_length}]"
        )
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"WikiText asset parent directory does not exist: {destination.parent}"
        )
    temporary_path = _save_npy_temporary(destination, array)
    try:
        _publish_npy_temporary_no_clobber(temporary_path, destination)
        return destination
    finally:
        temporary_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class TrainingAssets:
    """Read-only source memmaps plus the pilot's explicit CPU block selections."""

    manifest: dict[str, object]
    train_mmap: np.memmap
    validation_mmap: np.memmap
    test_mmap: np.memmap
    train_blocks: np.ndarray
    validation_blocks: np.ndarray
    train_permutation_seed: int
    train_permutation_sha256: str
    validation_block_indices: tuple[int, ...]

    @classmethod
    def load(cls, config: PilotConfig) -> "TrainingAssets":
        """Mmap assets and select train/validation blocks under the fixed protocol."""
        config.validate()
        manifest_path = data_manifest_path(config.data_root, config.sequence_length)
        identity_manifest = _read_data_manifest_structure(manifest_path)
        _validate_manifest_model_binding(
            identity_manifest,
            model_profile=config.resolved_model_profile,
            model_source=config.model_path,
        )
        _validate_manifest_tokenizer_files_for_model(config, identity_manifest)
        manifest = validate_data_manifest(
            manifest_path,
            sequence_length=config.sequence_length,
            include_piqa=False,
            model_profile=config.resolved_model_profile,
            model_source=config.model_path,
        )
        root = Path(config.data_root)
        train_mmap = _load_int32_blocks(
            wikitext_asset_path(root, "train", config.sequence_length),
            config.sequence_length,
        )
        validation_mmap = _load_int32_blocks(
            wikitext_asset_path(root, "validation", config.sequence_length),
            config.sequence_length,
        )
        test_mmap = _load_int32_blocks(
            wikitext_asset_path(root, "test", config.sequence_length),
            config.sequence_length,
        )
        wikitext = manifest["wikitext"]
        assert isinstance(wikitext, Mapping)
        train_record = wikitext["train"]
        assert isinstance(train_record, Mapping)
        permutations = train_record["permutations_by_master_seed"]
        assert isinstance(permutations, Mapping)
        selected = permutations[str(config.seed)]
        assert isinstance(selected, Mapping)
        permutation = _validate_permutation(
            selected["permutation"], train_mmap.shape[0]
        )
        permutation_seed = selected["permutation_seed"]
        permutation_sha256 = selected["permutation_sha256"]
        if type(permutation_seed) is not int or not isinstance(permutation_sha256, str):
            raise ValueError("selected train permutation identity is invalid")
        if config.train_blocks > train_mmap.shape[0]:
            raise ValueError(
                f"data asset has {train_mmap.shape[0]} train blocks, but config requires {config.train_blocks}"
            )
        if config.validation_blocks > validation_mmap.shape[0]:
            raise ValueError(
                "data asset has insufficient validation blocks for configured canonical validation"
            )
        train_blocks = np.array(
            train_mmap[permutation[: config.train_blocks]], dtype=np.int32, copy=True
        )
        validation_blocks = np.array(
            validation_mmap[: config.validation_blocks], dtype=np.int32, copy=True
        )
        return cls(
            manifest=manifest,
            train_mmap=train_mmap,
            validation_mmap=validation_mmap,
            test_mmap=test_mmap,
            train_blocks=train_blocks,
            validation_blocks=validation_blocks,
            train_permutation_seed=permutation_seed,
            train_permutation_sha256=permutation_sha256,
            validation_block_indices=tuple(range(config.validation_blocks)),
        )

    @staticmethod
    def load_piqa_rows(config: PilotConfig) -> tuple[dict[str, object], ...]:
        """Load PIQA only for the separately authorised secondary endpoint.

        NLL is the primary irreversible stage.  Its asset bundle must never
        touch PIQA, even incidentally, so the secondary rows are opened only
        after the formal paired-NLL gate has been revalidated by orchestration.
        """
        config.validate()
        validate_data_manifest(
            data_manifest_path(config.data_root, config.sequence_length),
            sequence_length=config.sequence_length,
            include_piqa=True,
        )
        return tuple(load_piqa_rows(piqa_asset_path(Path(config.data_root))))

    @staticmethod
    def collate(blocks: np.ndarray) -> dict[str, np.ndarray]:
        """Copy causal-LM input IDs and labels independently on the CPU.

        The training loop owns any subsequent ``torch.long`` CUDA transfer;
        this data layer intentionally performs no torch import or device move.
        """
        source = np.asarray(blocks)
        if source.dtype != np.int32 or source.ndim != 2:
            raise ValueError("collate expects an int32 [batch, sequence] block array")
        return {
            "input_ids": np.array(source, dtype=np.int32, copy=True),
            "labels": np.array(source, dtype=np.int32, copy=True),
        }


def _validate_manifest_tokenizer_files_for_model(
    config: PilotConfig, manifest: Mapping[str, object]
) -> None:
    """Bind newly profiled token assets to the tokenizer files at model assembly."""
    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise ValueError("data manifest tokenizer identity is invalid")
    for field, filename in (
        ("vocab_sha256", "tokenizer.json"),
        ("config_sha256", "tokenizer_config.json"),
    ):
        candidate = Path(config.model_path) / filename
        if not candidate.is_file():
            raise FileNotFoundError(
                f"configured model is missing tokenizer identity file: {candidate}"
            )
        if sha256_file(candidate) != tokenizer.get(field):
            raise ValueError(
                f"configured model tokenizer identity does not match prepared token assets: {filename}"
            )


def _read_data_manifest_structure(path: Path) -> dict[str, object]:
    """Read and structure-check a manifest without touching referenced block assets."""
    if not path.is_file():
        raise FileNotFoundError(f"data manifest does not exist: {path}")
    try:
        with path.open(encoding="utf-8") as manifest_file:
            raw = json.load(manifest_file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"data manifest JSON is invalid: {exc.msg}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("data manifest must be a JSON object")
    return _validate_manifest_structure(raw)


def _validate_manifest_model_binding(
    manifest: Mapping[str, object],
    *,
    model_profile: str | None,
    model_source: str | Path | None,
) -> None:
    """Compare model facts before opening any packed token block."""
    declared_profile = manifest.get("model_profile")
    declared_source = manifest.get("model_source")
    if model_profile is not None:
        if declared_profile is None:
            if model_profile != "qwen2_5_0_5b":
                raise ValueError(
                    "legacy data manifest is interpretable only as qwen2_5_0_5b"
                )
        elif declared_profile != model_profile:
            raise ValueError(
                "manifest model_profile does not match requested model_profile"
            )
    if model_source is not None and declared_source is not None:
        if declared_source != str(Path(model_source).resolve()):
            raise ValueError(
                "manifest model_source does not match requested model source"
            )


def formal_training_selection_evidence(
    config: PilotConfig, assets: TrainingAssets
) -> dict[str, object]:
    """Cryptographically bind native loaded blocks to schema-v2 selection facts.

    This deliberately reads the native manifest and source memmaps again rather
    than trusting attributes exposed by an injected ``assets`` lookalike.  It
    is the sole production bridge from immutable prepared data to checkpoint
    identity evidence.
    """
    if not isinstance(config, PilotConfig):
        raise TypeError("config must be a PilotConfig")
    if not isinstance(assets, TrainingAssets):
        raise TypeError("formal data-selection evidence requires native TrainingAssets")
    config.validate()
    manifest = validate_data_manifest(
        data_manifest_path(config.data_root, config.sequence_length),
        sequence_length=config.sequence_length,
        include_piqa=False,
    )
    if canonical_json(manifest) != canonical_json(assets.manifest):
        raise ValueError(
            "native TrainingAssets manifest does not match the validated manifest"
        )
    wikitext = manifest["wikitext"]
    assert isinstance(wikitext, Mapping)
    train_record = wikitext["train"]
    assert isinstance(train_record, Mapping)
    schedule = train_record["permutations_by_master_seed"]
    assert isinstance(schedule, Mapping)
    selected = schedule[str(config.seed)]
    assert isinstance(selected, Mapping)
    root = Path(config.data_root)
    source_train = _load_int32_blocks(
        wikitext_asset_path(root, "train", config.sequence_length),
        config.sequence_length,
    )
    source_validation = _load_int32_blocks(
        wikitext_asset_path(root, "validation", config.sequence_length),
        config.sequence_length,
    )
    permutation = _validate_permutation(selected["permutation"], source_train.shape[0])
    permutation_seed = selected["permutation_seed"]
    permutation_sha = selected["permutation_sha256"]
    expected_seed = config.seed_derivations.data_permutation_seed
    if type(permutation_seed) is not int or permutation_seed != expected_seed:
        raise ValueError(
            "native manifest selected permutation seed does not match formal config"
        )
    if type(permutation_sha) is not str or permutation_sha != permutation_sha256(
        permutation
    ):
        raise ValueError("native manifest selected permutation SHA-256 is invalid")
    expected_train = np.ascontiguousarray(
        np.asarray(source_train[permutation[: config.train_blocks]], dtype=np.int32)
    )
    actual_train = np.ascontiguousarray(np.asarray(assets.train_blocks))
    if (
        actual_train.dtype != np.int32
        or actual_train.shape != expected_train.shape
        or (not np.array_equal(actual_train, expected_train))
    ):
        raise ValueError(
            "native TrainingAssets train blocks do not match the selected permutation"
        )
    expected_validation_indices = tuple(range(config.validation_blocks))
    expected_validation = np.ascontiguousarray(
        np.asarray(source_validation[: config.validation_blocks], dtype=np.int32)
    )
    actual_validation = np.ascontiguousarray(np.asarray(assets.validation_blocks))
    if (
        actual_validation.dtype != np.int32
        or actual_validation.shape != expected_validation.shape
        or (not np.array_equal(actual_validation, expected_validation))
    ):
        raise ValueError(
            "native TrainingAssets validation blocks do not match canonical selection"
        )
    validation_indices = list(expected_validation_indices)
    train_blocks_sha = hashlib.sha256(actual_train.tobytes()).hexdigest()
    validation_blocks_sha = hashlib.sha256(actual_validation.tobytes()).hexdigest()
    validation_indices_sha = hashlib.sha256(
        canonical_json(validation_indices).encode("utf-8")
    ).hexdigest()
    selection_identity = hashlib.sha256(
        canonical_json(
            {
                "master_seed": config.seed,
                "permutation_seed": permutation_seed,
                "permutation_sha256": permutation_sha,
                "selected_train_blocks_sha256": train_blocks_sha,
            }
        ).encode("utf-8")
    ).hexdigest()
    return {
        "train_permutation_master_seed": config.seed,
        "train_permutation_seed": permutation_seed,
        "train_permutation_sha256": permutation_sha,
        "train_selected_blocks_sha256": train_blocks_sha,
        "train_permutation_identity": selection_identity,
        "validation_block_indices": validation_indices,
        "validation_block_indices_sha256": validation_indices_sha,
        "validation_selected_blocks_sha256": validation_blocks_sha,
    }


load_training_assets = TrainingAssets.load


def _validate_legacy_v1_manifest_structure(
    raw: Mapping[str, object],
) -> dict[str, object]:
    """Validate the retired v1 schema solely for an explicit migration.

    Normal readers intentionally never call this function: v1 cannot provide
    the seed schedule needed for a formal run.  Keeping this validation here
    makes the conversion prove both legacy provenance and local asset identity
    before it produces a v2 manifest.
    """
    manifest = dict(raw)
    required = {
        "schema_version",
        "kind",
        "sequence_length",
        "tokenizer",
        "rules",
        "wikitext",
        "piqa",
        "files",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(
            f"data manifest is missing required field(s): {', '.join(missing)}"
        )
    if manifest["schema_version"] != 1:
        raise ValueError("only data manifest schema_version 1 can be migrated")
    if manifest["kind"] != "qwen_lora_prepared_data":
        raise ValueError("data manifest kind is invalid")
    manifest["sequence_length"] = require_supported_sequence_length(
        manifest["sequence_length"]
    )
    manifest["tokenizer"] = _validate_tokenizer_identity(manifest["tokenizer"])
    if manifest["rules"] != _RULES:
        raise ValueError(
            "data manifest rules do not match the fixed preprocessing protocol"
        )
    wikitext = manifest["wikitext"]
    if not isinstance(wikitext, Mapping) or set(wikitext) != set(WIKITEXT_SPLITS):
        raise ValueError(
            "data manifest wikitext must contain exactly train, validation, and test"
        )
    normalized_wikitext: dict[str, dict[str, object]] = {}
    for split in WIKITEXT_SPLITS:
        split_record = wikitext[split]
        if not isinstance(split_record, Mapping):
            raise ValueError(f"data manifest wikitext.{split} must be an object")
        required_split = {
            "dataset_name",
            "dataset_config",
            "fingerprint",
            "revision",
            "block_count",
            "packed_token_count",
            "filename",
        }
        if split == "train":
            required_split |= {"permutation", "permutation_sha256", "permutation_seed"}
        missing_split = sorted(required_split - set(split_record))
        if missing_split:
            raise ValueError(
                f"data manifest wikitext.{split} is missing required field(s): {', '.join(missing_split)}"
            )
        if (
            type(split_record["block_count"]) is not int
            or split_record["block_count"] < 0
        ):
            raise ValueError(
                f"data manifest wikitext.{split}.block_count must be non-negative"
            )
        if (
            type(split_record["packed_token_count"]) is not int
            or split_record["packed_token_count"] < 0
        ):
            raise ValueError(
                f"data manifest wikitext.{split}.packed_token_count must be non-negative"
            )
        if not isinstance(split_record["filename"], str):
            raise ValueError(
                f"data manifest wikitext.{split}.filename must be a string"
            )
        if split_record["dataset_name"] != WIKITEXT_DATASET_NAME:
            raise ValueError(
                f"data manifest wikitext.{split}.dataset_name must be {WIKITEXT_DATASET_NAME!r}"
            )
        if split_record["dataset_config"] != WIKITEXT_DATASET_CONFIG:
            raise ValueError(
                f"data manifest wikitext.{split}.dataset_config must be {WIKITEXT_DATASET_CONFIG!r}"
            )
        _require_nonempty_string(
            split_record["fingerprint"], f"wikitext.{split}.fingerprint"
        )
        _require_nonempty_string(split_record["revision"], f"wikitext.{split}.revision")
        normalized_wikitext[split] = dict(split_record)
    train_record = normalized_wikitext["train"]
    train_block_count = train_record["block_count"]
    assert type(train_block_count) is int
    permutation = _validate_permutation(train_record["permutation"], train_block_count)
    expected_legacy = deterministic_permutation(train_block_count)
    if not np.array_equal(permutation, expected_legacy):
        raise ValueError("legacy train permutation does not match the seed-42 schedule")
    if train_record["permutation_seed"] != 42:
        raise ValueError("legacy data manifest train permutation_seed must be 42")
    if train_record["permutation_sha256"] != permutation_sha256(permutation):
        raise ValueError(
            "legacy data manifest train permutation_sha256 does not match permutation"
        )
    manifest["wikitext"] = normalized_wikitext
    piqa = manifest["piqa"]
    required_piqa = {
        "dataset_name",
        "split",
        "fingerprint",
        "revision",
        "filename",
        "row_count",
    }
    if not isinstance(piqa, Mapping) or set(piqa) != required_piqa:
        raise ValueError(
            "data manifest piqa must contain dataset_name, split, fingerprint, revision, filename, and row_count"
        )
    if (
        piqa["dataset_name"] != PIQA_DATASET_NAME
        or piqa["split"] != PIQA_VALIDATION_SPLIT
        or piqa["filename"] != PIQA_FILENAME
        or (type(piqa["row_count"]) is not int)
        or (piqa["row_count"] < 0)
    ):
        raise ValueError("data manifest piqa metadata is invalid")
    _require_nonempty_string(piqa["fingerprint"], "piqa.fingerprint")
    _require_nonempty_string(piqa["revision"], "piqa.revision")
    manifest["piqa"] = dict(piqa)
    files = manifest["files"]
    if not isinstance(files, Mapping):
        raise ValueError("data manifest files must be an object")
    expected_files = {PIQA_FILENAME} | {
        wikitext_asset_path(Path("."), split, manifest["sequence_length"]).name
        for split in WIKITEXT_SPLITS
    }
    if set(files) != expected_files:
        raise ValueError(
            "data manifest files do not match the canonical asset filenames"
        )
    normalized_files: dict[str, dict[str, object]] = {}
    for filename, record in files.items():
        if not isinstance(filename, str) or not isinstance(record, Mapping):
            raise ValueError("data manifest files records are invalid")
        _validate_file_record(filename, record)
        normalized_files[filename] = dict(record)
    manifest["files"] = normalized_files
    return manifest


def _upgrade_legacy_v1_manifest(legacy: Mapping[str, object]) -> dict[str, object]:
    """Derive the v2 seed schedule from a validated legacy manifest."""
    upgraded = dict(legacy)
    upgraded["schema_version"] = DATA_MANIFEST_SCHEMA_VERSION
    wikitext = legacy["wikitext"]
    assert isinstance(wikitext, Mapping)
    normalized_wikitext = {split: dict(wikitext[split]) for split in WIKITEXT_SPLITS}
    train = normalized_wikitext["train"]
    block_count = train["block_count"]
    assert type(block_count) is int
    train.pop("permutation")
    train.pop("permutation_sha256")
    train.pop("permutation_seed")
    train["permutations_by_master_seed"] = _build_train_permutations_by_master_seed(
        block_count, supplied=None
    )
    upgraded["wikitext"] = normalized_wikitext
    return _validate_manifest_structure(upgraded)


def _validate_manifest_structure(raw: Mapping[str, object]) -> dict[str, object]:
    manifest = dict(raw)
    required = {
        "schema_version",
        "kind",
        "sequence_length",
        "tokenizer",
        "rules",
        "wikitext",
        "piqa",
        "files",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(
            f"data manifest is missing required field(s): {', '.join(missing)}"
        )
    if manifest["schema_version"] == 1:
        raise ValueError(
            "data manifest schema_version 1 is retired; run scripts/migrate_data_manifest_v1.py --config <pilot-config.json> before retrying"
        )
    if manifest["schema_version"] != DATA_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"data manifest schema_version must be {DATA_MANIFEST_SCHEMA_VERSION}"
        )
    if manifest["kind"] != "qwen_lora_prepared_data":
        raise ValueError("data manifest kind is invalid")
    manifest["sequence_length"] = require_supported_sequence_length(
        manifest["sequence_length"]
    )
    manifest["tokenizer"] = _validate_tokenizer_identity(manifest["tokenizer"])
    optional_model_fields = {"model_profile", "model_source", "special_token_policy"}
    present_model_fields = optional_model_fields & set(manifest)
    if present_model_fields and present_model_fields != optional_model_fields:
        missing_model_fields = sorted(optional_model_fields - present_model_fields)
        raise ValueError(
            "data manifest model identity is missing required field(s): "
            + ", ".join(missing_model_fields)
        )
    if present_model_fields:
        from .backbones import geometry_for_profile

        profile = manifest["model_profile"]
        if not isinstance(profile, str):
            raise ValueError("data manifest model_profile must be a string")
        geometry_for_profile(profile)
        source = manifest["model_source"]
        if (
            not isinstance(source, str)
            or not source
            or source != str(Path(source).resolve())
        ):
            raise ValueError(
                "data manifest model_source must be an absolute normalized path"
            )
        if manifest["special_token_policy"] != NO_SPECIAL_TOKEN_POLICY:
            raise ValueError(
                "data manifest special_token_policy does not match the no-extra-BOS/EOS protocol"
            )
    if manifest["rules"] != _RULES:
        raise ValueError(
            "data manifest rules do not match the fixed preprocessing protocol"
        )
    wikitext = manifest["wikitext"]
    if not isinstance(wikitext, Mapping):
        raise ValueError("data manifest wikitext must be an object")
    if set(wikitext) != set(WIKITEXT_SPLITS):
        raise ValueError(
            "data manifest wikitext must contain exactly train, validation, and test"
        )
    normalized_wikitext: dict[str, dict[str, object]] = {}
    for split in WIKITEXT_SPLITS:
        split_record = wikitext[split]
        if not isinstance(split_record, Mapping):
            raise ValueError(f"data manifest wikitext.{split} must be an object")
        required_split = {
            "dataset_name",
            "dataset_config",
            "fingerprint",
            "revision",
            "block_count",
            "packed_token_count",
            "filename",
        }
        if split == "train":
            required_split |= {"permutations_by_master_seed"}
        missing_split = sorted(required_split - set(split_record))
        if missing_split:
            raise ValueError(
                f"data manifest wikitext.{split} is missing required field(s): {', '.join(missing_split)}"
            )
        if (
            type(split_record["block_count"]) is not int
            or split_record["block_count"] < 0
        ):
            raise ValueError(
                f"data manifest wikitext.{split}.block_count must be non-negative"
            )
        if (
            type(split_record["packed_token_count"]) is not int
            or split_record["packed_token_count"] < 0
        ):
            raise ValueError(
                f"data manifest wikitext.{split}.packed_token_count must be non-negative"
            )
        if not isinstance(split_record["filename"], str):
            raise ValueError(
                f"data manifest wikitext.{split}.filename must be a string"
            )
        if split_record["dataset_name"] != WIKITEXT_DATASET_NAME:
            raise ValueError(
                f"data manifest wikitext.{split}.dataset_name must be {WIKITEXT_DATASET_NAME!r}"
            )
        if split_record["dataset_config"] != WIKITEXT_DATASET_CONFIG:
            raise ValueError(
                f"data manifest wikitext.{split}.dataset_config must be {WIKITEXT_DATASET_CONFIG!r}"
            )
        _require_nonempty_string(
            split_record["fingerprint"], f"wikitext.{split}.fingerprint"
        )
        _require_nonempty_string(split_record["revision"], f"wikitext.{split}.revision")
        normalized_wikitext[split] = dict(split_record)
    train_record = normalized_wikitext["train"]
    train_block_count = train_record["block_count"]
    assert type(train_block_count) is int
    _validate_train_permutations_by_master_seed(
        train_record["permutations_by_master_seed"], train_block_count
    )
    manifest["wikitext"] = normalized_wikitext
    piqa = manifest["piqa"]
    required_piqa = {
        "dataset_name",
        "split",
        "fingerprint",
        "revision",
        "filename",
        "row_count",
    }
    if not isinstance(piqa, Mapping) or set(piqa) != required_piqa:
        raise ValueError(
            "data manifest piqa must contain dataset_name, split, fingerprint, revision, filename, and row_count"
        )
    if (
        piqa["dataset_name"] != PIQA_DATASET_NAME
        or piqa["split"] != PIQA_VALIDATION_SPLIT
        or piqa["filename"] != PIQA_FILENAME
        or (type(piqa["row_count"]) is not int)
        or (piqa["row_count"] < 0)
    ):
        raise ValueError("data manifest piqa metadata is invalid")
    _require_nonempty_string(piqa["fingerprint"], "piqa.fingerprint")
    _require_nonempty_string(piqa["revision"], "piqa.revision")
    manifest["piqa"] = dict(piqa)
    files = manifest["files"]
    if not isinstance(files, Mapping):
        raise ValueError("data manifest files must be an object")
    expected_files = {PIQA_FILENAME} | {
        wikitext_asset_path(Path("."), split, manifest["sequence_length"]).name
        for split in WIKITEXT_SPLITS
    }
    if set(files) != expected_files:
        raise ValueError(
            "data manifest files do not match the canonical asset filenames"
        )
    normalized_files: dict[str, dict[str, object]] = {}
    for filename, record in files.items():
        if not isinstance(filename, str) or not isinstance(record, Mapping):
            raise ValueError("data manifest files records are invalid")
        _validate_file_record(filename, record)
        normalized_files[filename] = dict(record)
    manifest["files"] = normalized_files
    return manifest


def _validate_wikitext_metadata(
    wikitext: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    if not isinstance(wikitext, Mapping) or set(wikitext) != set(WIKITEXT_SPLITS):
        raise ValueError(
            "wikitext metadata must contain exactly train, validation, and test"
        )
    normalized: dict[str, dict[str, object]] = {}
    for split in WIKITEXT_SPLITS:
        metadata = wikitext[split]
        if not isinstance(metadata, Mapping):
            raise ValueError(f"wikitext metadata for {split} must be an object")
        if "fingerprint" not in metadata or "revision" not in metadata:
            raise ValueError(
                f"wikitext metadata for {split} requires fingerprint and revision"
            )
        fingerprint = metadata["fingerprint"]
        revision = metadata["revision"]
        _require_nonempty_string(
            fingerprint, f"wikitext metadata for {split}.fingerprint"
        )
        _require_nonempty_string(revision, f"wikitext metadata for {split}.revision")
        normalized[split] = {"fingerprint": fingerprint, "revision": revision}
    return normalized


def _validate_piqa_provenance(provenance: Mapping[str, object]) -> dict[str, str]:
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "fingerprint",
        "revision",
    }:
        raise ValueError(
            "PIQA provenance must contain exactly fingerprint and revision"
        )
    fingerprint = _require_nonempty_string(
        provenance["fingerprint"], "PIQA provenance.fingerprint"
    )
    revision = _require_nonempty_string(
        provenance["revision"], "PIQA provenance.revision"
    )
    return {"fingerprint": fingerprint, "revision": revision}


def _validate_tokenizer_identity(tokenizer: Mapping[str, object]) -> dict[str, object]:
    normalized = _validate_json_object(tokenizer, "tokenizer")
    if not normalized:
        raise ValueError("tokenizer identity must be non-empty")
    _require_nonempty_string(normalized.get("name_or_path"), "tokenizer.name_or_path")
    for field in ("vocab_sha256", "config_sha256"):
        digest = normalized.get(field)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any((character not in "0123456789abcdef" for character in digest))
        ):
            raise ValueError(f"tokenizer.{field} must be a lowercase SHA-256 digest")
    return normalized


def _file_record(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"data asset does not exist: {path}")
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def _validate_file_record(filename: str, record: Mapping[str, object]) -> None:
    if set(record) != {"sha256", "bytes"}:
        raise ValueError(
            f"data manifest file record for {filename} must contain sha256 and bytes"
        )
    digest = record["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any((character not in "0123456789abcdef" for character in digest))
    ):
        raise ValueError(f"data manifest SHA-256 is invalid for {filename}")
    if type(record["bytes"]) is not int or record["bytes"] < 0:
        raise ValueError(f"data manifest byte count is invalid for {filename}")


def _load_int32_blocks(path: Path, sequence_length: int) -> np.memmap:
    if not path.is_file():
        raise FileNotFoundError(f"WikiText asset does not exist: {path}")
    try:
        blocks = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"WikiText asset could not be loaded: {path}") from exc
    if not isinstance(blocks, np.memmap):
        raise ValueError(
            f"WikiText asset must load as a memory-mapped NPY file: {path}"
        )
    if (
        blocks.dtype != np.int32
        or blocks.ndim != 2
        or blocks.shape[1] != sequence_length
    ):
        raise ValueError(
            f"WikiText asset must be int32 with shape [num_blocks, {sequence_length}]: {path}"
        )
    return blocks


def _validate_permutation(
    values: Iterable[int] | np.ndarray, block_count: int
) -> np.ndarray:
    if type(block_count) is not int or block_count < 0:
        raise ValueError("block_count must be non-negative")
    if isinstance(values, np.ndarray):
        raw_values = values.tolist()
    else:
        try:
            raw_values = list(values)
        except TypeError as exc:
            raise ValueError("permutation must be an iterable of integers") from exc
    if any(
        (
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            for value in raw_values
        )
    ):
        raise ValueError(
            "permutation must contain integer values, not bool, float, or string"
        )
    try:
        permutation = np.asarray(raw_values, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError("permutation must contain integers") from exc
    if permutation.ndim != 1 or len(permutation) != block_count:
        raise ValueError("permutation length must equal train block_count")
    if not np.array_equal(np.sort(permutation), np.arange(block_count, dtype=np.int64)):
        raise ValueError(
            "permutation must contain every train block index exactly once"
        )
    return permutation


def _build_train_permutations_by_master_seed(
    block_count: int, *, supplied: Mapping[str, Iterable[int] | np.ndarray] | None
) -> dict[str, dict[str, object]]:
    """Build the complete formal schedule in one manifest generation."""
    if supplied is not None and (not isinstance(supplied, Mapping)):
        raise ValueError("train_permutations_by_master_seed must be a mapping")
    if supplied is not None and set(supplied) != {
        str(seed) for seed in FORMAL_MASTER_SEEDS
    }:
        raise ValueError(
            "train_permutations_by_master_seed must contain exactly the formal master seeds"
        )
    records: dict[str, dict[str, object]] = {}
    for master_seed in FORMAL_MASTER_SEEDS:
        derivations = resolve_seed_derivations(master_seed)
        permutation = (
            deterministic_permutation(
                block_count, seed=derivations.data_permutation_seed
            )
            if supplied is None
            else _validate_permutation(supplied[str(master_seed)], block_count)
        )
        expected = deterministic_permutation(
            block_count, seed=derivations.data_permutation_seed
        )
        if not np.array_equal(permutation, expected):
            raise ValueError(
                "train permutation must match its derived formal permutation seed"
            )
        records[str(master_seed)] = {
            "permutation": permutation.astype(np.int64).tolist(),
            "permutation_sha256": permutation_sha256(permutation),
            "permutation_seed": derivations.data_permutation_seed,
        }
    return records


def _validate_train_permutations_by_master_seed(
    value: object, block_count: int
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        str(seed) for seed in FORMAL_MASTER_SEEDS
    }:
        raise ValueError(
            "data manifest train permutations_by_master_seed must contain exactly 17, 42, and 73"
        )
    for master_seed in FORMAL_MASTER_SEEDS:
        field = f"data manifest train permutations_by_master_seed.{master_seed}"
        record = value[str(master_seed)]
        if not isinstance(record, Mapping) or set(record) != {
            "permutation",
            "permutation_sha256",
            "permutation_seed",
        }:
            raise ValueError(
                f"{field} must contain permutation, permutation_sha256, and permutation_seed"
            )
        permutation = _validate_permutation(record["permutation"], block_count)
        derivations = resolve_seed_derivations(master_seed)
        if record["permutation_seed"] != derivations.data_permutation_seed:
            raise ValueError(
                f"{field}.permutation_seed does not match the formal seed protocol"
            )
        if record["permutation_sha256"] != permutation_sha256(permutation):
            raise ValueError(f"{field}.permutation_sha256 does not match permutation")
        expected = deterministic_permutation(
            block_count, seed=derivations.data_permutation_seed
        )
        if not np.array_equal(permutation, expected):
            raise ValueError(
                f"{field}.permutation does not match the derived seed order"
            )


def _save_npy_temporary(destination: Path, array: np.ndarray) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp.npy", dir=destination.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        np.save(temporary_path, array, allow_pickle=False)
        return temporary_path
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _publish_npy_temporary_no_clobber(temporary_path: Path, destination: Path) -> None:
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
