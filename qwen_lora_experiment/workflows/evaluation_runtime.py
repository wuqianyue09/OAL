"""Reconstruct the runtime and evidence shared by offline evaluation stages."""

from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from .. import run_artifacts
from . import prepare
from .. import assets as data_assets
from .. import config as config_module
from .. import paths as file_paths
from .. import telemetry
from .. import checkpoint_identity
from .. import runtime_identity
from .. import runtime_execution
from ..assets import validate_data_manifest
from ..config import PilotConfig
from .evaluate_common import _require_evaluation_mode
from ..workflows.errors import OrchestrationError
from ..workflows.prepare import match_run_effective_config


@dataclass(frozen=True)
class _PreparedOfflineEvaluation:
    """Runtime and immutable evidence shared by the separately published stages."""

    run_dir: Path
    store: telemetry.RunStateStore | None
    assets: object
    runtime: object
    checkpoint_context: object
    attention_execution: dict[str, object]
    execution: dict[str, object]
    data_identity: dict[str, object]
    model_identity: dict[str, object]
    source_identity: dict[str, object]
    experiment_identity: dict[str, object]
    preloaded_piqa_rows: object | None


def _prepare_offline_evaluation(
    config: PilotConfig,
    *,
    run_dir: str | Path,
    assets_loader: Callable[[PilotConfig], object] | None = None,
    model_builder: Callable[[PilotConfig, object], object] | None = None,
    model_identity_collector: (
        Callable[[str | Path], Mapping[str, object]] | None
    ) = None,
    manifest_validator: Callable[..., dict[str, object]] = validate_data_manifest,
    source_paths: Sequence[str | Path] | None = None,
    device: object | None = None,
    allow_source_drift: bool = False,
    nonformal_test_mode: bool = False,
    evaluation_mode: str = "pilot",
    preloaded_piqa_rows: object | None = None,
    read_only_sidecar: bool = False,
    hd_manifest_path: str | Path | None = None,
    hd_probe_path: str | Path | None = None,
) -> _PreparedOfflineEvaluation:
    """Rebuild one offline bundle while preserving stage-independent evidence.

    Ordinary experiment-source drift is recorded as compatibility evidence,
    never used as a repository-wide evaluation gate.  Model/data/config and
    any public-kernel admission evidence remain semantic identities and are
    checked by the callers that consume this setup.
    """
    if not isinstance(config, PilotConfig):
        raise TypeError("config must be a PilotConfig")
    if type(allow_source_drift) is not bool:
        raise TypeError("allow_source_drift must be a boolean")
    if type(nonformal_test_mode) is not bool:
        raise TypeError("nonformal_test_mode must be a boolean")
    if type(read_only_sidecar) is not bool:
        raise TypeError("read_only_sidecar must be a boolean")
    _require_evaluation_mode(evaluation_mode)
    config.validate()
    injected_runtime_seam = any(
        (
            candidate is not None
            for candidate in (assets_loader, model_builder, model_identity_collector)
        )
    )
    if injected_runtime_seam and (not nonformal_test_mode):
        raise OrchestrationError(
            "injected evaluation dependencies require nonformal_test_mode=True"
        )
    production_evaluation = not nonformal_test_mode
    destination = Path(run_dir)
    store = (
        None
        if read_only_sidecar
        else telemetry.RunStateStore(destination / "status.json")
    )
    runtime_preparation_started = False
    try:
        if not destination.is_dir():
            raise FileNotFoundError(
                f"evaluation run directory does not exist: {destination}"
            )
        if not read_only_sidecar:
            assert store is not None
            status = store.read()
            if status.get("state") != "trained":
                raise OrchestrationError(
                    f"evaluation requires run status trained; found {status.get('state')!r}"
                )
        stored_config = (
            config_module.from_mapping(
                run_artifacts._safe_run_artifact_json(
                    destination,
                    run_artifacts.EFFECTIVE_CONFIG_FILENAME,
                    "effective config",
                )
            )
            if production_evaluation
            else config_module.from_json_file(
                destination / run_artifacts.EFFECTIVE_CONFIG_FILENAME
            )
        )
        config = match_run_effective_config(config, stored_config)
        stored_manifest = (
            run_artifacts._safe_run_artifact_json(
                destination,
                run_artifacts.DATA_MANIFEST_COPY_FILENAME,
                "data-manifest evidence",
            )
            if production_evaluation
            else run_artifacts._read_json_mapping(
                destination / run_artifacts.DATA_MANIFEST_COPY_FILENAME,
                "evaluation data-manifest evidence",
            )
        )
        if manifest_validator is validate_data_manifest:
            current_manifest = manifest_validator(
                data_assets.data_manifest_path(
                    config.data_root, config.sequence_length
                ),
                sequence_length=config.sequence_length,
                include_piqa=False,
            )
        else:
            current_manifest = manifest_validator(
                data_assets.data_manifest_path(
                    config.data_root, config.sequence_length
                ),
                sequence_length=config.sequence_length,
            )
        if not isinstance(current_manifest, Mapping):
            raise OrchestrationError("data manifest validator must return a mapping")
        if file_paths.canonical_json(stored_manifest) != file_paths.canonical_json(
            dict(current_manifest)
        ):
            raise OrchestrationError(
                "evaluation data manifest does not match immutable run evidence"
            )
        stored_hashes = run_artifacts._read_json_mapping(
            destination / run_artifacts.SOURCE_HASHES_FILENAME,
            "evaluation source-hash evidence",
        )
        training_source_hashes = run_artifacts._validated_source_hash_mapping(
            stored_hashes.get("files"), "evaluation training source hashes"
        )
        resolved_sources = (
            prepare._default_source_paths(config)
            if source_paths is None
            else tuple(source_paths)
        )
        current_hashes = file_paths.source_file_hashes(resolved_sources)
        source_drift = prepare._source_hash_drift(
            training_source_hashes, current_hashes
        )
        evaluation_source_identity = prepare._evaluation_source_identity(
            destination,
            training_source_hashes=training_source_hashes,
            evaluation_source_hashes=current_hashes,
            source_drift=source_drift,
            allow_source_drift=allow_source_drift,
        )
        if assets_loader is None:
            from ..assets import TrainingAssets

            assets_loader = TrainingAssets.load
        if model_builder is None:
            from ..model_setup import build_model_bundle

            def default_model_builder(profile: PilotConfig, target: object) -> object:
                return build_model_bundle(
                    profile,
                    device=target,
                    hd_manifest_path=hd_manifest_path,
                    hd_probe_path=hd_probe_path,
                )

            model_builder = default_model_builder
        if model_identity_collector is None:
            from ..paths import collect_local_model_identity

            model_identity_collector = collect_local_model_identity
        if not all(
            (
                callable(candidate)
                for candidate in (
                    assets_loader,
                    model_builder,
                    model_identity_collector,
                )
            )
        ):
            raise TypeError("evaluation runtime dependencies must be callable")
        resolved_device = (
            runtime_execution._production_cuda_device() if device is None else device
        )
        runtime_preparation_started = True
        assets = assets_loader(config)
        assets_manifest = getattr(assets, "manifest", None)
        if not isinstance(assets_manifest, Mapping):
            raise OrchestrationError(
                "assets loader must return an object with a manifest mapping"
            )
        if file_paths.canonical_json(
            dict(assets_manifest)
        ) != file_paths.canonical_json(stored_manifest):
            raise OrchestrationError(
                "loaded evaluation assets do not match immutable run evidence"
            )
        identity = runtime_identity._stable_model_identity(
            model_identity_collector(config.model_path)
        )
        runtime = model_builder(config, resolved_device)
        required_runtime_fields = (
            "model",
            "tokenizer",
            "kernel_bank",
            "device",
            "attention_execution",
        )
        if any((not hasattr(runtime, name) for name in required_runtime_fields)):
            raise OrchestrationError(
                "model builder must return runtime with model, tokenizer, kernel_bank, device, and attention_execution"
            )
        required_asset_fields = ("validation_mmap", "test_mmap")
        if any((not hasattr(assets, name) for name in required_asset_fields)):
            raise OrchestrationError(
                "assets loader must provide validation_mmap and test_mmap"
            )
        runtime_identity._assert_runtime_tokenizer_matches_manifest(
            config, runtime, stored_manifest
        )
        context_data_identity = checkpoint_identity._checkpoint_data_identity(
            config, run_dir=destination, assets=assets, formal=production_evaluation
        )
        attention_execution = runtime_execution._runtime_attention_execution(
            config, runtime
        )
        from ..checkpointing import add_model_scope_identity

        topology_identity, identity, attention_execution = add_model_scope_identity(
            config,
            config_identity={"method_identity": config.method_identity},
            model_identity=identity,
            attention_execution=attention_execution,
            kernel_bank=runtime.kernel_bank,
            allow_missing_kernel_scope=not production_evaluation,
        )
        from ..checkpointing import CheckpointContext

        protocol_identity = (
            runtime_identity._frozen_protocol_identity(destination)
            if production_evaluation
            else None
        )
        checkpoint_config_identity, identity, attention_execution = (
            add_model_scope_identity(
                config,
                config_identity=checkpoint_identity._checkpoint_config_identity(
                    config,
                    destination,
                    formal=production_evaluation,
                    runtime_evidence_sha256=(
                        run_artifacts._safe_run_artifact_sha256(
                            destination, run_artifacts.RUNTIME_EVIDENCE_FILENAME
                        )
                        if production_evaluation
                        else None
                    ),
                    lora_initial_identity=checkpoint_identity._load_or_create_lora_initial_hashes_identity(
                        destination,
                        model=None,
                        formal=production_evaluation,
                        create=False,
                    ),
                    protocol_identity=protocol_identity,
                ),
                model_identity=identity,
                attention_execution=attention_execution,
                kernel_bank=runtime.kernel_bank,
                allow_missing_kernel_scope=not production_evaluation,
            )
        )
        context = CheckpointContext(
            method=config.method,
            config_identity=checkpoint_config_identity,
            data_identity=context_data_identity,
            model_identity=identity,
        )
        execution = {
            "entrypoint": "scripts/evaluate_pilot.py",
            "evaluation_scope": "pilot_single_seed",
            "device": str(runtime.device),
            "sequence_length": config.sequence_length,
            "model_loader": "local_files_only",
            "checkpoint_policy": "best_adapter_only",
            "attention": attention_execution,
        }
        from .evaluate_common import model_scope_execution

        execution.update(model_scope_execution(config))
        data_identity = {
            "manifest_sha256": (
                run_artifacts._safe_run_artifact_sha256(
                    destination, run_artifacts.DATA_MANIFEST_COPY_FILENAME
                )
                if production_evaluation
                else file_paths.sha256_file(
                    destination / run_artifacts.DATA_MANIFEST_COPY_FILENAME
                )
            ),
            "current_manifest_sha256": file_paths.sha256_file(
                data_assets.data_manifest_path(config.data_root, config.sequence_length)
            ),
            "sequence_length": config.sequence_length,
        }
        experiment_identity = runtime_identity._experiment_identity(
            config,
            runtime=runtime,
            attention_execution=attention_execution,
            effective_config_sha256=(
                run_artifacts._safe_run_artifact_sha256(
                    destination, run_artifacts.EFFECTIVE_CONFIG_FILENAME
                )
                if production_evaluation
                else file_paths.sha256_file(
                    destination / run_artifacts.EFFECTIVE_CONFIG_FILENAME
                )
            ),
            data_manifest_sha256=data_identity["manifest_sha256"],
            model_identity=identity,
            data_manifest=stored_manifest,
        )
        experiment_identity.update(topology_identity)
        experiment_identity["attention_execution"] = dict(attention_execution)
        experiment_identity["model_identity"] = dict(identity)
        if production_evaluation:
            experiment_identity = runtime_identity._adopt_frozen_protocol_identity(
                destination, experiment_identity
            )
            runtime_identity._require_runtime_evidence(destination, experiment_identity)
        return _PreparedOfflineEvaluation(
            run_dir=destination,
            store=store,
            assets=assets,
            runtime=runtime,
            checkpoint_context=context,
            attention_execution=attention_execution,
            execution=execution,
            data_identity=data_identity,
            model_identity=identity,
            source_identity=evaluation_source_identity,
            experiment_identity=experiment_identity,
            preloaded_piqa_rows=preloaded_piqa_rows,
        )
    except BaseException as exc:
        if runtime_preparation_started and store is not None:
            prepare._record_preparation_failure(store, exc)
        raise
