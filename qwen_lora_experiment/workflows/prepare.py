"""Model-free data and durable run preparation for pilot workflows."""

from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import json
from pathlib import Path
from .. import data as data_module
from .. import run_artifacts
from ..assets import data_manifest_path, validate_data_manifest
from ..config import PilotConfig, from_json_file
from .errors import OrchestrationError, PreflightFailure
from ..paths import (
    canonical_json,
    collect_preflight,
    sha256_bytes,
    sha256_file,
    source_file_hashes,
)
from ..telemetry import RunStateSink, RunStateStore
from ..run_artifacts import (
    DATA_MANIFEST_COPY_FILENAME,
    EFFECTIVE_CONFIG_FILENAME,
    PREFLIGHT_FILENAME,
    SOURCE_HASHES_FILENAME,
)


@dataclass(frozen=True)
class PreparedPilotRun:
    """Durable, model-free evidence prepared before a runtime may be built."""

    run_dir: Path
    config: PilotConfig
    data_manifest: dict[str, object]
    preflight: dict[str, object]
    source_hashes: dict[str, str]
    status_store: RunStateSink


def prepare_pilot_data(
    config: PilotConfig,
    *,
    wikitext_revision: str = "main",
    piqa_revision: str = "main",
    tokenizer_loader: Callable[[Path], object] | None = None,
    dataset_downloader: (
        Callable[..., tuple[Mapping[str, object], object]] | None
    ) = None,
) -> dict[str, object]:
    """Build the immutable data-root asset set for one validated sequence length.

    ``scripts/prepare_data.py`` supplies the only permitted downloader. All
    output writers are no-clobber: a repeat is accepted only when every
    canonical byte and manifest fact already agrees with this request.
    """
    if not isinstance(config, PilotConfig):
        raise TypeError("config must be a PilotConfig")
    config.validate()
    _require_nonempty_revision(wikitext_revision, "wikitext_revision")
    _require_nonempty_revision(piqa_revision, "piqa_revision")
    from ..assets import (
        _read_data_manifest_structure,
        build_data_manifest,
        data_manifest_path,
        validate_data_manifest,
        write_data_manifest,
    )
    from ..data import (
        WIKITEXT_SPLITS,
        dataset_split_identity,
        piqa_asset_path,
        serialize_piqa_rows,
        tokenize_wikitext_rows,
        tokenizer_identity,
        wikitext_asset_path,
        write_packed_token_blocks,
    )

    root = Path(config.data_root)
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise NotADirectoryError(f"data_root is not a directory: {root}")
    manifest_path = data_manifest_path(root, config.sequence_length)
    if manifest_path.exists():
        identity_manifest = _read_data_manifest_structure(manifest_path)
        _assert_local_tokenizer_files_match_manifest(config, identity_manifest)
        existing_manifest = validate_data_manifest(
            manifest_path, sequence_length=config.sequence_length
        )
        return _prepared_data_summary(root, manifest_path, existing_manifest)
    existing_n_assets = [
        wikitext_asset_path(root, split, config.sequence_length)
        for split in WIKITEXT_SPLITS
        if wikitext_asset_path(root, split, config.sequence_length).exists()
    ]
    if existing_n_assets:
        rendered = ", ".join((str(path) for path in existing_n_assets))
        raise OrchestrationError(
            f"refusing to download into an incomplete immutable asset set without its manifest: {rendered}"
        )
    if dataset_downloader is None:
        raise TypeError(
            "prepare_pilot_data requires an explicit dataset_downloader from prepare_data.py"
        )
    if tokenizer_loader is None:
        tokenizer_loader = _load_local_pilot_tokenizer
    if not callable(tokenizer_loader) or not callable(dataset_downloader):
        raise TypeError("tokenizer_loader and dataset_downloader must be callable")
    tokenizer = tokenizer_loader(Path(config.model_path))
    tokenizer_record = tokenizer_identity(
        tokenizer,
        vocab_file=data_module._local_tokenizer_file(
            config.model_path, "tokenizer.json"
        ),
        config_file=data_module._local_tokenizer_file(
            config.model_path, "tokenizer_config.json"
        ),
    )
    downloaded = dataset_downloader(
        wikitext_revision=wikitext_revision, piqa_revision=piqa_revision
    )
    if not isinstance(downloaded, tuple) or len(downloaded) != 2:
        raise TypeError(
            "dataset_downloader must return (wikitext_splits, piqa_validation)"
        )
    wikitext_splits, piqa_validation = downloaded
    if not isinstance(wikitext_splits, Mapping) or set(wikitext_splits) != set(
        WIKITEXT_SPLITS
    ):
        raise ValueError(
            "dataset_downloader must return exactly train, validation, and test splits"
        )
    wikitext_metadata: dict[str, dict[str, object]] = {}
    for split in WIKITEXT_SPLITS:
        rows = wikitext_splits[split]
        provenance = dataset_split_identity(rows, revision=wikitext_revision)
        _require_dataset_provenance(provenance, f"WikiText {split}")
        wikitext_metadata[split] = provenance
        write_packed_token_blocks(
            tokenize_wikitext_rows(rows, tokenizer),
            wikitext_asset_path(root, split, config.sequence_length),
            sequence_length=config.sequence_length,
        )
    piqa_provenance = dataset_split_identity(piqa_validation, revision=piqa_revision)
    _require_dataset_provenance(piqa_provenance, "PIQA validation")
    serialize_piqa_rows(piqa_validation, piqa_asset_path(root))
    manifest = build_data_manifest(
        sequence_length=config.sequence_length,
        tokenizer=tokenizer_record,
        model_profile=config.resolved_model_profile,
        model_source=config.model_path,
        wikitext=wikitext_metadata,
        piqa_provenance=piqa_provenance,
        data_root=root,
    )
    manifest_path = write_data_manifest(manifest_path, manifest)
    validated_manifest = validate_data_manifest(
        manifest_path,
        sequence_length=config.sequence_length,
        tokenizer=tokenizer_record,
    )
    return _prepared_data_summary(root, manifest_path, validated_manifest)


def _prepared_data_summary(
    data_root: Path, manifest_path: Path, manifest: Mapping[str, object]
) -> dict[str, object]:
    """Return only facts already verified by an immutable data manifest."""
    wikitext = manifest["wikitext"]
    piqa = manifest["piqa"]
    if not isinstance(wikitext, Mapping) or not isinstance(piqa, Mapping):
        raise OrchestrationError("validated data manifest has malformed split records")
    block_counts: dict[str, object] = {}
    for split in ("train", "validation", "test"):
        split_record = wikitext.get(split)
        if not isinstance(split_record, Mapping) or "block_count" not in split_record:
            raise OrchestrationError(
                f"validated data manifest is missing {split} block_count"
            )
        block_counts[split] = split_record["block_count"]
    if "sequence_length" not in manifest or "row_count" not in piqa:
        raise OrchestrationError("validated data manifest is missing summary facts")
    return {
        "data_root": str(data_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "sequence_length": manifest["sequence_length"],
        "wikitext_block_counts": block_counts,
        "piqa_row_count": piqa["row_count"],
    }


def _load_local_pilot_tokenizer(model_path: Path) -> object:
    """Load a tokenizer from local model files only, after argument parsing."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)


def _assert_local_tokenizer_files_match_manifest(
    config: PilotConfig, manifest: Mapping[str, object]
) -> None:
    """Verify local tokenizer payloads before reusing an immutable block set."""
    expected = manifest.get("tokenizer")
    if not isinstance(expected, Mapping):
        raise OrchestrationError("prepared data manifest is missing tokenizer identity")
    declared_profile = manifest.get("model_profile")
    if declared_profile is None:
        if config.resolved_model_profile != "qwen2_5_0_5b":
            raise OrchestrationError(
                "legacy prepared data manifest is interpretable only as qwen2_5_0_5b"
            )
    elif declared_profile != config.resolved_model_profile:
        raise OrchestrationError(
            "prepared data manifest model_profile does not match the configured model"
        )
    declared_source = manifest.get("model_source")
    if declared_source is not None and declared_source != str(
        Path(config.model_path).resolve()
    ):
        raise OrchestrationError(
            "prepared data manifest model source does not match the configured model"
        )
    for field, filename in (
        ("vocab_sha256", "tokenizer.json"),
        ("config_sha256", "tokenizer_config.json"),
    ):
        expected_hash = expected.get(field)
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise OrchestrationError(
                f"prepared data manifest tokenizer {field} is invalid"
            )
        actual_hash = sha256_file(
            data_module._local_tokenizer_file(config.model_path, filename)
        )
        if actual_hash != expected_hash:
            raise OrchestrationError(
                f"local tokenizer identity does not match prepared data manifest: {filename}"
            )


def _require_nonempty_revision(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_dataset_provenance(provenance: Mapping[str, object], label: str) -> None:
    fingerprint = provenance.get("fingerprint")
    revision = provenance.get("revision")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(f"{label} dataset fingerprint must be a non-empty string")
    if not isinstance(revision, str) or not revision:
        raise ValueError(f"{label} dataset revision must be a non-empty string")


def prepare_pilot_run(
    config: PilotConfig,
    *,
    run_dir: str | Path,
    require_cuda: bool = False,
    require_bf16: bool = False,
    require_oal_attention: bool = False,
    manifest_validator: Callable[..., dict[str, object]] = validate_data_manifest,
    preflight_collector: Callable[..., Mapping[str, object]] = collect_preflight,
    source_paths: Sequence[str | Path] | None = None,
) -> PreparedPilotRun:
    """Create one immutable run directory and record inputs before model setup.

    This function only validates local files and process capabilities. It
    deliberately never builds a model, touches tokenizer/model weights, or
    invokes a CUDA kernel. Any failure after status creation is converted into
    the run's structured ``failed`` status before being raised to the caller.
    """
    if not isinstance(config, PilotConfig):
        raise TypeError("config must be a PilotConfig")
    config.validate()
    destination = Path(run_dir)
    _create_empty_run_directory(destination)
    store = RunStateStore(destination / "status.json")
    status_store = store
    status_store.create(run_id=destination.name, method=config.method, run_kind="pilot")
    try:
        run_artifacts._write_mutable_run_evidence(
            destination, EFFECTIVE_CONFIG_FILENAME, config.to_dict()
        )
        manifest = manifest_validator(
            data_manifest_path(config.data_root, config.sequence_length),
            sequence_length=config.sequence_length,
        )
        if not isinstance(manifest, Mapping):
            raise OrchestrationError("data manifest validator must return a mapping")
        copied_manifest = dict(manifest)
        run_artifacts._write_mutable_run_evidence(
            destination, DATA_MANIFEST_COPY_FILENAME, copied_manifest
        )
        resolved_sources = (
            _default_source_paths(config)
            if source_paths is None
            else tuple(source_paths)
        )
        hashes = source_file_hashes(resolved_sources)
        run_artifacts._write_mutable_run_evidence(
            destination, SOURCE_HASHES_FILENAME, {"files": hashes}
        )
        preflight = preflight_collector(
            require_cuda=require_cuda,
            require_bf16=require_bf16,
            require_oal_attention=require_oal_attention,
        )
        if not isinstance(preflight, Mapping):
            raise OrchestrationError("preflight collector must return a mapping")
        copied_preflight = dict(preflight)
        run_artifacts._write_mutable_run_evidence(
            destination, PREFLIGHT_FILENAME, copied_preflight
        )
        if copied_preflight.get("ok") is not True:
            raise PreflightFailure(
                "read-only preflight reported missing required capabilities"
            )
        return PreparedPilotRun(
            run_dir=destination,
            config=config,
            data_manifest=copied_manifest,
            preflight=copied_preflight,
            source_hashes=hashes,
            status_store=status_store,
        )
    except BaseException as exc:
        _record_preparation_failure(status_store, exc)
        raise


def prepare_pilot_resume(
    config: PilotConfig,
    *,
    run_dir: str | Path,
    require_cuda: bool = True,
    require_bf16: bool = True,
    require_oal_attention: bool = False,
    manifest_validator: Callable[..., dict[str, object]] = validate_data_manifest,
    preflight_collector: Callable[..., Mapping[str, object]] = collect_preflight,
    source_paths: Sequence[str | Path] | None = None,
) -> PreparedPilotRun:
    """Reopen one interrupted run only after re-checking immutable evidence."""
    if not isinstance(config, PilotConfig):
        raise TypeError("config must be a PilotConfig")
    config.validate()
    destination = Path(run_dir)
    if not destination.is_dir():
        raise FileNotFoundError(f"resume run directory does not exist: {destination}")
    store = RunStateStore(destination / "status.json")
    status_store = store
    if status_store is not store:
        status_store.read()
    store.reopen_for_resume()
    try:
        stored_config = resolve_run_effective_config(config, destination)
        manifest = manifest_validator(
            data_manifest_path(stored_config.data_root, stored_config.sequence_length),
            sequence_length=stored_config.sequence_length,
        )
        if not isinstance(manifest, Mapping):
            raise OrchestrationError("data manifest validator must return a mapping")
        copied_manifest = dict(manifest)
        stored_manifest_path = destination / DATA_MANIFEST_COPY_FILENAME
        if not stored_manifest_path.is_file():
            raise FileNotFoundError(
                f"resume is missing data-manifest copy: {stored_manifest_path}"
            )
        try:
            stored_manifest = json.loads(
                stored_manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise OrchestrationError("resume data-manifest copy is unreadable") from exc
        if canonical_json(stored_manifest) != canonical_json(copied_manifest):
            raise OrchestrationError(
                "resume data manifest does not match immutable run evidence"
            )
        resolved_sources = (
            _default_source_paths(stored_config)
            if source_paths is None
            else tuple(source_paths)
        )
        hashes = source_file_hashes(resolved_sources)
        stored_hashes_path = destination / SOURCE_HASHES_FILENAME
        try:
            stored_hashes = json.loads(stored_hashes_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OrchestrationError(
                "resume source-hash evidence is unreadable"
            ) from exc
        if not isinstance(stored_hashes, Mapping):
            raise OrchestrationError("resume source-hash evidence is malformed")
        training_hashes = run_artifacts._validated_source_hash_mapping(
            stored_hashes.get("files"), "resume training source hashes"
        )
        run_artifacts._write_mutable_run_evidence(
            destination,
            "resume_source_identity.json",
            _evaluation_source_identity(
                destination,
                training_source_hashes=training_hashes,
                evaluation_source_hashes=hashes,
                source_drift=_source_hash_drift(training_hashes, hashes),
                allow_source_drift=True,
            ),
        )
        preflight = preflight_collector(
            require_cuda=require_cuda,
            require_bf16=require_bf16,
            require_oal_attention=require_oal_attention,
        )
        if not isinstance(preflight, Mapping):
            raise OrchestrationError("preflight collector must return a mapping")
        copied_preflight = dict(preflight)
        run_artifacts._write_mutable_run_evidence(
            destination, "resume_preflight.json", copied_preflight
        )
        if copied_preflight.get("ok") is not True:
            raise PreflightFailure(
                "read-only resume preflight reported missing required capabilities"
            )
        return PreparedPilotRun(
            run_dir=destination,
            config=stored_config,
            data_manifest=copied_manifest,
            preflight=copied_preflight,
            source_hashes=hashes,
            status_store=status_store,
        )
    except BaseException as exc:
        _record_preparation_failure(status_store, exc)
        raise


def resolve_run_effective_config(
    requested_config: PilotConfig, run_dir: str | Path
) -> PilotConfig:
    """Return the persisted run config after one narrow legacy-Qwen interpretation."""
    if not isinstance(requested_config, PilotConfig):
        raise TypeError("requested_config must be a PilotConfig")
    path = Path(run_dir) / EFFECTIVE_CONFIG_FILENAME
    stored_config = from_json_file(path)
    return match_run_effective_config(requested_config, stored_config)


def match_run_effective_config(
    requested_config: PilotConfig, stored_config: PilotConfig
) -> PilotConfig:
    """Compare requested and persisted configs under the same legacy interpretation."""
    if not isinstance(requested_config, PilotConfig) or not isinstance(
        stored_config, PilotConfig
    ):
        raise TypeError("requested_config and stored_config must be PilotConfig values")
    comparable_stored = stored_config
    comparable_requested = requested_config
    if stored_config.model_profile is None or requested_config.model_profile is None:
        legacy_source = (
            stored_config if stored_config.model_profile is None else requested_config
        )
        _require_actual_legacy_qwen_model(legacy_source.model_path)
        if stored_config.model_profile is None:
            comparable_stored = replace(stored_config, model_profile="qwen2_5_0_5b")
        if requested_config.model_profile is None:
            comparable_requested = replace(
                requested_config, model_profile="qwen2_5_0_5b"
            )
    if comparable_stored != comparable_requested:
        raise OrchestrationError(
            "requested config does not exactly match effective_config.json"
        )
    return stored_config


def _require_actual_legacy_qwen_model(model_path: str | Path) -> None:
    config_path = Path(model_path) / "config.json"
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OrchestrationError(
            "legacy Qwen effective config requires a readable local model config"
        ) from exc
    expected = {
        "model_type": "qwen2",
        "num_hidden_layers": 24,
        "hidden_size": 896,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
    }
    if (
        not isinstance(raw, Mapping)
        or any((raw.get(name) != value for (name, value) in expected.items()))
        or ("head_dim" in raw and raw["head_dim"] != 64)
    ):
        raise OrchestrationError(
            "profile-less effective config is legacy Qwen only, but local model_type/geometry does not prove Qwen2.5-0.5B"
        )


def _source_hash_drift(
    training_source_hashes: Mapping[str, str],
    evaluation_source_hashes: Mapping[str, str],
) -> list[dict[str, object]]:
    """Return the exact sorted training/evaluator source difference."""
    drift: list[dict[str, object]] = []
    for path in sorted(set(training_source_hashes) | set(evaluation_source_hashes)):
        training_digest = training_source_hashes.get(path)
        evaluation_digest = evaluation_source_hashes.get(path)
        if training_digest == evaluation_digest:
            continue
        drift.append(
            {
                "path": path,
                "training_sha256": training_digest,
                "evaluation_sha256": evaluation_digest,
            }
        )
    return drift


def _evaluation_source_identity(
    run_dir: Path,
    *,
    training_source_hashes: Mapping[str, str],
    evaluation_source_hashes: Mapping[str, str],
    source_drift: Sequence[Mapping[str, object]],
    allow_source_drift: bool,
) -> dict[str, object]:
    """Persist separate snapshots and a non-semantic source-drift warning."""
    evaluation_record = {"files": dict(evaluation_source_hashes)}
    evaluation_record_sha256 = sha256_bytes(
        (canonical_json(evaluation_record) + "\n").encode("utf-8")
    )
    return {
        "source_hashes_sha256": sha256_file(run_dir / SOURCE_HASHES_FILENAME),
        "source_hashes": dict(training_source_hashes),
        "evaluation_source_hashes_sha256": evaluation_record_sha256,
        "evaluation_source_hashes": dict(evaluation_source_hashes),
        "source_drift_allowed": True,
        "legacy_source_drift_recovery_requested": allow_source_drift,
        "source_compatibility_policy": "record_nonsemantic_source_drift_only",
        "source_drift": [dict(item) for item in source_drift],
    }


def _create_empty_run_directory(destination: Path) -> None:
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to reuse existing run directory: {destination}"
        ) from exc
    if not destination.is_dir():
        raise NotADirectoryError(f"run_dir is not a directory: {destination}")


def _record_preparation_failure(store: RunStateSink, error: BaseException) -> None:
    try:
        store.transition("failed", failure=error)
    except (OSError, ValueError):
        pass


def _default_source_paths(config: PilotConfig | None = None) -> tuple[Path, ...]:
    """Record the source files shipped with this OAL release."""
    if config is not None and (not isinstance(config, PilotConfig)):
        raise TypeError("config must be a PilotConfig or None")
    project_root = Path(__file__).resolve().parents[2]
    paths = []
    for folder in ("qwen_lora_experiment", "oal_attention", "scripts"):
        paths.extend(
            (
                p
                for p in (project_root / folder).rglob("*")
                if p.is_file() and p.suffix in {".py", ".cpp", ".json"}
            )
        )
    paths.append(project_root / "configs/lora_qwen_n4096_template.json")
    return tuple(sorted(paths))
