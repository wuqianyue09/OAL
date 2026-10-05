"""Training workflow entry point, importable without the legacy facade."""

from __future__ import annotations
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from .. import checkpoint_identity
from .. import run_artifacts
from .. import runtime_identity
from .. import runtime_execution
from ..config import PilotConfig
from ..paths import canonical_json, sha256_file
from .errors import OrchestrationError
from .prepare import (
    PreparedPilotRun,
    prepare_pilot_run,
    prepare_pilot_resume,
    _record_preparation_failure,
)


@dataclass(frozen=True)
class _TrainingExecutionResult:
    """Workflow result and the data evidence consumed by this execution."""

    result_record: dict[str, object]
    data_manifest: dict[str, object]


def execute_pilot_training(
    config: PilotConfig,
    *,
    run_dir: str | Path,
    resume: bool,
    prepared_run: PreparedPilotRun | None = None,
    assets_loader: Callable[[PilotConfig], object] | None = None,
    model_builder: Callable[[PilotConfig, object], object] | None = None,
    model_identity_collector: (
        Callable[[str | Path], Mapping[str, object]] | None
    ) = None,
    resume_loader: Callable[..., object] | None = None,
    training_runner: Callable[..., object] | None = None,
    device: object | None = None,
    hd_manifest_path: str | Path | None = None,
    hd_probe_path: str | Path | None = None,
    nonformal_test_mode: bool = False,
) -> dict[str, object]:
    """Run a pilot and return the existing public training result."""
    return _execute_training(
        config,
        run_dir=run_dir,
        resume=resume,
        prepared_run=prepared_run,
        assets_loader=assets_loader,
        model_builder=model_builder,
        model_identity_collector=model_identity_collector,
        resume_loader=resume_loader,
        training_runner=training_runner,
        device=device,
        hd_manifest_path=hd_manifest_path,
        hd_probe_path=hd_probe_path,
        nonformal_test_mode=nonformal_test_mode,
    ).result_record


def _execute_training(
    config: PilotConfig,
    *,
    run_dir: str | Path,
    resume: bool,
    prepared_run: PreparedPilotRun | None = None,
    assets_loader: Callable[[PilotConfig], object] | None = None,
    model_builder: Callable[[PilotConfig, object], object] | None = None,
    model_identity_collector: (
        Callable[[str | Path], Mapping[str, object]] | None
    ) = None,
    resume_loader: Callable[..., object] | None = None,
    training_runner: Callable[..., object] | None = None,
    device: object | None = None,
    hd_manifest_path: str | Path | None = None,
    hd_probe_path: str | Path | None = None,
    nonformal_test_mode: bool = False,
    measurement_runtime: object | None = None,
    admitted_grouped_request: object | None = None,
) -> _TrainingExecutionResult:
    """Prepare evidence, assemble the model, and execute normal or measured training."""
    if not isinstance(config, PilotConfig):
        raise TypeError("config must be a PilotConfig")
    if type(nonformal_test_mode) is not bool:
        raise TypeError("nonformal_test_mode must be a boolean")
    if prepared_run is not None and (not nonformal_test_mode):
        raise OrchestrationError(
            "caller-supplied prepared_run requires nonformal_test_mode=True; it cannot create formal runtime or checkpoint evidence"
        )
    injected_runtime_seam = any(
        (
            candidate is not None
            for candidate in (
                assets_loader,
                model_builder,
                model_identity_collector,
                resume_loader,
                training_runner,
            )
        )
    )
    if injected_runtime_seam and (not nonformal_test_mode):
        raise OrchestrationError(
            "injected runtime dependencies require nonformal_test_mode=True; they cannot create formal checkpoint or runtime evidence"
        )
    formal_execution = not injected_runtime_seam and (not nonformal_test_mode)
    config.validate()
    if measurement_runtime is not None and (
        not formal_execution or config.method != "grouped_quadratic"
    ):
        raise OrchestrationError(
            "training measurement requires formal grouped execution"
        )
    formal_grouped_admission: object | None = None
    if prepared_run is None:
        operator_required = runtime_execution._method_requires_oal_attention(
            config.method
        )
        if resume:
            prepared = prepare_pilot_resume(
                config,
                run_dir=run_dir,
                require_cuda=True,
                require_bf16=True,
                require_oal_attention=operator_required,
            )
        else:
            prepared = prepare_pilot_run(
                config,
                run_dir=run_dir,
                require_cuda=True,
                require_bf16=True,
                require_oal_attention=operator_required,
            )
    else:
        prepared = prepared_run
        if prepared.config != config or prepared.run_dir != Path(run_dir):
            raise OrchestrationError(
                "prepared run does not match the requested config and run_dir"
            )
    config = prepared.config
    try:
        if assets_loader is None:
            from ..assets import TrainingAssets

            assets_loader = TrainingAssets.load
            use_native_block_sources = True
        else:
            use_native_block_sources = False
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
        if resume_loader is None:
            from ..checkpointing import load_latest_resume

            resume_loader = load_latest_resume
        if training_runner is None:
            from ..training import run_training

            training_runner = run_training
        resolved_device = (
            runtime_execution._production_cuda_device() if device is None else device
        )
        assets = assets_loader(config)
        manifest = getattr(assets, "manifest", None)
        if not isinstance(manifest, Mapping):
            raise OrchestrationError(
                "assets loader must return an object with a manifest mapping"
            )
        if canonical_json(dict(manifest)) != canonical_json(prepared.data_manifest):
            raise OrchestrationError(
                "loaded training assets do not match immutable run evidence"
            )
        identity = runtime_identity._stable_model_identity(
            model_identity_collector(config.model_path)
        )
        historical_model_identity = dict(identity)
        runtime = model_builder(config, resolved_device)
        required_runtime_fields = ("model", "kernel_bank", "device")
        required_runtime_fields += ("optimizer", "scheduler")
        if any((not hasattr(runtime, name) for name in required_runtime_fields)):
            if config.tuning_mode == "lora":
                raise OrchestrationError(
                    "model builder must return runtime with model, kernel_bank, optimizer, scheduler, and device"
                )
            raise OrchestrationError(
                "model builder returned a runtime missing policy-required fields"
            )
        runtime_identity._assert_runtime_tokenizer_matches_manifest(
            config, runtime, prepared.data_manifest
        )
        attention_execution = runtime_execution._runtime_attention_execution(
            config, runtime
        )
        historical_attention_execution = dict(attention_execution)
        from ..checkpointing import add_model_scope_identity

        topology_identity, identity, attention_execution = add_model_scope_identity(
            config,
            config_identity={"method_identity": config.method_identity},
            model_identity=identity,
            attention_execution=attention_execution,
            kernel_bank=runtime.kernel_bank,
            allow_missing_kernel_scope=not formal_execution,
        )
        if formal_execution:
            runtime_evidence = runtime_identity._experiment_identity(
                config,
                runtime=runtime,
                attention_execution=attention_execution,
                effective_config_sha256=sha256_file(
                    prepared.run_dir / run_artifacts.EFFECTIVE_CONFIG_FILENAME
                ),
                data_manifest_sha256=sha256_file(
                    prepared.run_dir / run_artifacts.DATA_MANIFEST_COPY_FILENAME
                ),
                model_identity=historical_model_identity,
                data_manifest=prepared.data_manifest,
                formal_grouped_admission=None,
                _formal_grouped_evidence_handoff=runtime_identity._FORMAL_GROUPED_EVIDENCE_HANDOFF,
            )
            runtime_evidence.update(topology_identity)
            runtime_evidence["attention_execution"] = dict(attention_execution)
            runtime_evidence["model_identity"] = dict(identity)
            runtime_identity._write_or_validate_runtime_evidence(
                prepared.run_dir, runtime_evidence
            )
        from ..telemetry import JsonlWriter

        runtime_evidence_sha256 = (
            None
            if not formal_execution
            else sha256_file(prepared.run_dir / run_artifacts.RUNTIME_EVIDENCE_FILENAME)
        )
        from ..checkpointing import CheckpointContext

        checkpoint_config_identity, identity, attention_execution = (
            add_model_scope_identity(
                config,
                config_identity=checkpoint_identity._checkpoint_config_identity(
                    config,
                    prepared.run_dir,
                    formal=formal_execution,
                    runtime_evidence_sha256=runtime_evidence_sha256,
                    lora_initial_identity=checkpoint_identity._load_or_create_lora_initial_hashes_identity(
                        prepared.run_dir,
                        model=runtime.model,
                        formal=formal_execution,
                        create=not resume,
                    ),
                ),
                model_identity=identity,
                attention_execution=attention_execution,
                kernel_bank=runtime.kernel_bank,
                allow_missing_kernel_scope=not formal_execution,
            )
        )
        context = CheckpointContext(
            method=config.method,
            config_identity=checkpoint_config_identity,
            data_identity=checkpoint_identity._checkpoint_data_identity(
                config, run_dir=prepared.run_dir, assets=assets, formal=formal_execution
            ),
            model_identity=dict(identity),
        )
        from ..protocol import preserve_global_rng_state

        with preserve_global_rng_state():
            resume_cursor: Mapping[str, object] | None = None
            if resume:
                assert resume_loader is not None
                restored = resume_loader(
                    prepared.run_dir,
                    model=runtime.model,
                    kernel_bank=runtime.kernel_bank,
                    context=context,
                    optimizer=runtime.optimizer,
                    scheduler=runtime.scheduler,
                )
                candidate_cursor = getattr(restored, "cursor", None)
                if not isinstance(candidate_cursor, Mapping):
                    raise OrchestrationError(
                        "strict resume loader must return a cursor mapping"
                    )
                resume_cursor = dict(candidate_cursor)
            if use_native_block_sources:
                train_blocks, validation_blocks = _native_training_block_sources(assets)
            else:
                train_blocks = getattr(assets, "train_blocks", None)
                validation_blocks = getattr(assets, "validation_blocks", None)
            if train_blocks is None or validation_blocks is None:
                raise OrchestrationError(
                    "assets loader must provide train_blocks and validation_blocks"
                )
            metrics_writer = JsonlWriter(prepared.run_dir / "metrics.jsonl")
            training_admission: dict[str, object] | None = None
            runner_arguments: dict[str, object] = {
                "config": config,
                "model": runtime.model,
                "kernel_bank": runtime.kernel_bank,
                "optimizer": runtime.optimizer,
                "train_blocks": train_blocks,
                "validation_blocks": validation_blocks,
                "run_dir": prepared.run_dir,
                "checkpoint_context": context,
                "metrics_writer": metrics_writer,
                "status_store": prepared.status_store,
                "device": runtime.device,
                "scheduler": runtime.scheduler,
                "resume_cursor": resume_cursor,
            }
            if measurement_runtime is not None:
                runner_arguments["measurement_runtime"] = measurement_runtime
            assert training_runner is not None
            result = training_runner(**runner_arguments)
        steps_completed = getattr(result, "steps_completed", None)
        next_train_block = getattr(result, "next_train_block", None)
        if type(steps_completed) is not int or type(next_train_block) is not int:
            raise OrchestrationError(
                "training runner must return integer steps_completed and next_train_block"
            )
        result_record = {
            "resumed": resume,
            "steps_completed": steps_completed,
            "next_train_block": next_train_block,
        }
        return _TrainingExecutionResult(result_record, dict(manifest))
    except BaseException as exc:
        _record_preparation_failure(prepared.status_store, exc)
        raise


def _native_training_block_sources(
    assets: object,
) -> tuple[Callable[[int], object], Callable[[], object]]:
    """Convert immutable NumPy blocks to explicit Torch inputs only at runtime."""
    import torch

    train_blocks = getattr(assets, "train_blocks", None)
    validation_blocks = getattr(assets, "validation_blocks", None)
    if train_blocks is None or validation_blocks is None:
        raise OrchestrationError("native training assets are missing block selections")

    def train_block_at(index: int) -> object:
        return {"input_ids": torch.as_tensor(train_blocks[index]).unsqueeze(0)}

    def validation_pass() -> object:
        return (
            {"input_ids": torch.as_tensor(block).unsqueeze(0)}
            for block in validation_blocks
        )

    return (train_block_at, validation_pass)
