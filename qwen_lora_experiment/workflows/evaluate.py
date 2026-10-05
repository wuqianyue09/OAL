"""Offline evaluation of a trained OAL Qwen LoRA adapter."""

from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
import time
from . import prepare
from ..assets import validate_data_manifest
from ..config import PilotConfig
from .evaluate_common import _require_evaluation_mode
from .evaluation_mmlu import (
    _grouped_base_mmlu_quality_execution,
    _load_pilot_mmlu_inputs,
    _mmlu_runtime_observation,
    _preflight_pilot_mmlu_request,
    _reset_mmlu_cuda_peak_memory,
    _run_grouped_base_mmlu_reference_smoke_and_alignment,
    _write_pilot_mmlu_orchestration_metadata,
)
from .evaluation_reference import (
    _grouped_piqa_candidate_lengths,
    _grouped_base_reference_session,
    _load_grouped_piqa_inputs,
    _piqa_reference_quality_execution,
    _require_grouped_piqa_reference_length_bound,
)
from .evaluation_runtime import _prepare_offline_evaluation
from .errors import OrchestrationError


def execute_pilot_nll_evaluation(
    config: PilotConfig,
    *,
    run_dir: str | Path,
    evaluation_mode: str = "pilot",
    assets_loader: Callable[[PilotConfig], object] | None = None,
    model_builder: Callable[[PilotConfig, object], object] | None = None,
    model_identity_collector: (
        Callable[[str | Path], Mapping[str, object]] | None
    ) = None,
    evaluation_runner: Callable[..., object] | None = None,
    best_loader: Callable[..., object] | None = None,
    manifest_validator: Callable[..., dict[str, object]] = validate_data_manifest,
    source_paths: Sequence[str | Path] | None = None,
    device: object | None = None,
    hd_manifest_path: str | Path | None = None,
    hd_probe_path: str | Path | None = None,
    allow_source_drift: bool = False,
    nonformal_test_mode: bool = False,
) -> object:
    """Publish validation/test NLL from this run's best adapter."""
    _require_evaluation_mode(evaluation_mode)
    _require_test_dependencies(
        nonformal_test_mode,
        assets_loader,
        model_builder,
        model_identity_collector,
        evaluation_runner,
        best_loader,
    )
    if not nonformal_test_mode:
        from ..evaluation import _run_pilot_nll_evaluation_stage

        evaluation_runner = _run_pilot_nll_evaluation_stage
    elif evaluation_runner is None:
        from ..evaluation import run_nll_evaluation_stage

        evaluation_runner = run_nll_evaluation_stage
    if best_loader is None:
        from ..checkpointing import load_best_adapter

        best_loader = load_best_adapter
    if not callable(evaluation_runner) or not callable(best_loader):
        raise TypeError(
            "NLL evaluation requires callable evaluation_runner and best_loader"
        )
    prepared = _prepare_offline_evaluation(
        config,
        run_dir=run_dir,
        assets_loader=assets_loader,
        model_builder=model_builder,
        model_identity_collector=model_identity_collector,
        manifest_validator=manifest_validator,
        source_paths=source_paths,
        device=device,
        allow_source_drift=allow_source_drift,
        nonformal_test_mode=nonformal_test_mode,
        evaluation_mode="pilot",
        hd_manifest_path=hd_manifest_path,
        hd_probe_path=hd_probe_path,
    )
    try:
        return evaluation_runner(
            prepared.run_dir,
            model=getattr(prepared.runtime, "model"),
            kernel_bank=getattr(prepared.runtime, "kernel_bank"),
            checkpoint_context=prepared.checkpoint_context,
            validation_blocks=getattr(prepared.assets, "validation_mmap"),
            test_blocks=getattr(prepared.assets, "test_mmap"),
            execution=prepared.execution,
            data_identity=prepared.data_identity,
            model_identity=prepared.model_identity,
            source_identity=prepared.source_identity,
            experiment_identity=prepared.experiment_identity,
            state_store=prepared.store,
            best_loader=best_loader,
            device=getattr(prepared.runtime, "device"),
            sequence_length=config.sequence_length,
        )
    except BaseException as exc:
        prepare._record_preparation_failure(prepared.store, exc)
        raise


def execute_pilot_piqa_evaluation(
    config: PilotConfig,
    *,
    run_dir: str | Path,
    evaluation_mode: str = "pilot",
    assets_loader: Callable[[PilotConfig], object] | None = None,
    model_builder: Callable[[PilotConfig, object], object] | None = None,
    model_identity_collector: (
        Callable[[str | Path], Mapping[str, object]] | None
    ) = None,
    evaluation_runner: Callable[..., object] | None = None,
    best_loader: Callable[..., object] | None = None,
    manifest_validator: Callable[..., dict[str, object]] = validate_data_manifest,
    source_paths: Sequence[str | Path] | None = None,
    device: object | None = None,
    hd_manifest_path: str | Path | None = None,
    hd_probe_path: str | Path | None = None,
    allow_source_drift: bool = False,
    nonformal_test_mode: bool = False,
) -> object:
    """Publish PIQA after this run's pilot NLL stage, using grouped reference attention."""
    _require_evaluation_mode(evaluation_mode)
    _require_test_dependencies(
        nonformal_test_mode,
        assets_loader,
        model_builder,
        model_identity_collector,
        evaluation_runner,
        best_loader,
    )
    if not nonformal_test_mode:
        from ..evaluation import _run_pilot_piqa_evaluation_stage

        evaluation_runner = _run_pilot_piqa_evaluation_stage
    elif evaluation_runner is None:
        from ..evaluation import run_piqa_evaluation_stage

        evaluation_runner = run_piqa_evaluation_stage
    if best_loader is None:
        from ..checkpointing import load_best_adapter

        best_loader = load_best_adapter
    if not callable(evaluation_runner) or not callable(best_loader):
        raise TypeError(
            "PIQA evaluation requires callable evaluation_runner and best_loader"
        )
    preloaded_piqa_rows = None
    grouped_piqa_reference_quality = not nonformal_test_mode
    if grouped_piqa_reference_quality:
        from ..evaluation_predictions import _load_nll_stage_record
        from ..evaluation_records import _require_pilot_nll_prerequisite

        _require_pilot_nll_prerequisite(
            _load_nll_stage_record(Path(run_dir) / "pilot_nll_eval.json")
        )
        tokenizer, preloaded_piqa_rows = _load_grouped_piqa_inputs(
            config, run_dir=Path(run_dir)
        )
        _require_grouped_piqa_reference_length_bound(
            _grouped_piqa_candidate_lengths(tokenizer, preloaded_piqa_rows)
        )
    prepared = _prepare_offline_evaluation(
        config,
        run_dir=run_dir,
        assets_loader=assets_loader,
        model_builder=model_builder,
        model_identity_collector=model_identity_collector,
        manifest_validator=manifest_validator,
        source_paths=source_paths,
        device=device,
        allow_source_drift=allow_source_drift,
        nonformal_test_mode=nonformal_test_mode,
        evaluation_mode="pilot",
        preloaded_piqa_rows=preloaded_piqa_rows,
        hd_manifest_path=hd_manifest_path,
        hd_probe_path=hd_probe_path,
    )
    try:
        piqa_rows = (
            prepared.preloaded_piqa_rows
            if prepared.preloaded_piqa_rows is not None
            else _load_piqa_rows_for_stage(config, prepared.assets)
        )
        execution = (
            _piqa_reference_quality_execution(config, prepared.execution)
            if grouped_piqa_reference_quality
            else prepared.execution
        )
        session = (
            _grouped_base_reference_session(
                config,
                getattr(prepared.runtime, "model"),
                expected_layer_ids=config.replacement_layer_ids,
            )
            if grouped_piqa_reference_quality
            else nullcontext()
        )
        with session:
            return evaluation_runner(
                prepared.run_dir,
                model=getattr(prepared.runtime, "model"),
                tokenizer=getattr(prepared.runtime, "tokenizer"),
                kernel_bank=getattr(prepared.runtime, "kernel_bank"),
                checkpoint_context=prepared.checkpoint_context,
                piqa_rows=piqa_rows,
                execution=execution,
                data_identity=prepared.data_identity,
                model_identity=prepared.model_identity,
                source_identity=prepared.source_identity,
                experiment_identity=prepared.experiment_identity,
                state_store=prepared.store,
                best_loader=best_loader,
                device=getattr(prepared.runtime, "device"),
            )
    except BaseException as exc:
        prepare._record_preparation_failure(prepared.store, exc)
        raise


def execute_pilot_mmlu_evaluation(
    config: PilotConfig,
    *,
    run_dir: str | Path,
    evaluation_mode: str = "pilot",
    assets_loader: Callable[[PilotConfig], object] | None = None,
    model_builder: Callable[[PilotConfig, object], object] | None = None,
    model_identity_collector: (
        Callable[[str | Path], Mapping[str, object]] | None
    ) = None,
    evaluation_runner: Callable[..., object] | None = None,
    best_loader: Callable[..., object] | None = None,
    manifest_validator: Callable[..., dict[str, object]] = validate_data_manifest,
    source_paths: Sequence[str | Path] | None = None,
    device: object | None = None,
    hd_manifest_path: str | Path | None = None,
    hd_probe_path: str | Path | None = None,
    allow_source_drift: bool = False,
    nonformal_test_mode: bool = False,
) -> object:
    """Publish the pilot-only, post-hoc MMLU sidecar without touching run status.

    Unlike NLL and PIQA, this endpoint has no lifecycle transition: it first
    binds the existing pilot NLL evidence to ``best_adapter.pt``, then opens
    MMLU's independently versioned bundle.  Runtime/scoring failures are
    intentionally propagated without status recovery or failure recording.
    """
    evaluation_mode = _require_evaluation_mode(evaluation_mode)
    if not isinstance(config, PilotConfig):
        raise TypeError("config must be a PilotConfig")
    config.validate()
    if type(nonformal_test_mode) is not bool:
        raise TypeError("nonformal_test_mode must be a boolean")
    injected_runtime_seam = any(
        (
            candidate is not None
            for candidate in (
                assets_loader,
                model_builder,
                model_identity_collector,
                evaluation_runner,
                best_loader,
            )
        )
    )
    if injected_runtime_seam and (not nonformal_test_mode):
        raise OrchestrationError(
            "injected MMLU evaluation dependencies require nonformal_test_mode=True; pilot MMLU evidence uses only production dependencies"
        )
    if not nonformal_test_mode:
        from ..evaluation import _run_pilot_mmlu_evaluation_stage

        evaluation_runner = _run_pilot_mmlu_evaluation_stage
    elif evaluation_runner is None:
        from ..evaluation import run_mmlu_evaluation_stage

        evaluation_runner = run_mmlu_evaluation_stage
    if best_loader is None:
        from ..checkpointing import load_best_adapter

        best_loader = load_best_adapter
    if not callable(evaluation_runner):
        raise TypeError("MMLU evaluation runner must be callable")
    if not callable(best_loader):
        raise TypeError("pilot MMLU evaluation requires a callable best_loader")
    _preflight_pilot_mmlu_request(config, run_dir=run_dir)
    (
        _mmlu_tokenizer,
        mmlu_dev_rows,
        mmlu_test_rows,
        expected_test_ids,
        mmlu_data_identity,
        mmlu_token_work,
    ) = _load_pilot_mmlu_inputs(config)
    grouped_mmlu_reference_quality = (
        not nonformal_test_mode and config.uses_grouped_quadratic_base
    )
    prepared = _prepare_offline_evaluation(
        config,
        run_dir=run_dir,
        assets_loader=assets_loader,
        model_builder=model_builder,
        model_identity_collector=model_identity_collector,
        manifest_validator=manifest_validator,
        source_paths=source_paths,
        device=device,
        allow_source_drift=allow_source_drift,
        nonformal_test_mode=nonformal_test_mode,
        evaluation_mode="pilot",
        read_only_sidecar=True,
        hd_manifest_path=hd_manifest_path,
        hd_probe_path=hd_probe_path,
    )
    execution = prepared.execution
    metadata: dict[str, object] = {
        "schema": "qwen_lora_pilot_mmlu_orchestration_metadata_v1",
        "token_work": mmlu_token_work,
    }
    session = (
        _grouped_base_reference_session(
            config,
            getattr(prepared.runtime, "model"),
            expected_layer_ids=config.replacement_layer_ids,
        )
        if grouped_mmlu_reference_quality
        else nullcontext()
    )
    _reset_mmlu_cuda_peak_memory(getattr(prepared.runtime, "device"))
    started_at = time.perf_counter()
    with session:
        if grouped_mmlu_reference_quality:
            grouped_observed = _run_grouped_base_mmlu_reference_smoke_and_alignment(
                config,
                model=getattr(prepared.runtime, "model"),
                kernel_bank=getattr(prepared.runtime, "kernel_bank"),
                tokenizer=getattr(prepared.runtime, "tokenizer"),
                dev_rows=mmlu_dev_rows,
                test_rows=mmlu_test_rows,
                layer_ids=config.replacement_layer_ids,
                device=getattr(prepared.runtime, "device"),
                seed=config.seed_derivations.diagnostic_inputs_seed,
            )
            disclosure = _grouped_base_mmlu_quality_execution(
                config,
                execution,
                token_work=mmlu_token_work,
                smoke=grouped_observed["smoke"],
                alignment=grouped_observed["alignment"],
            )
            quality_key = "grouped_mmlu_quality_execution"
            metadata[quality_key] = disclosure[quality_key]
        mmlu_data_identity = {**mmlu_data_identity, "orchestration_metadata": metadata}
        result = evaluation_runner(
            prepared.run_dir,
            model=getattr(prepared.runtime, "model"),
            tokenizer=getattr(prepared.runtime, "tokenizer"),
            kernel_bank=getattr(prepared.runtime, "kernel_bank"),
            checkpoint_context=prepared.checkpoint_context,
            mmlu_dev_rows=mmlu_dev_rows,
            mmlu_test_rows=mmlu_test_rows,
            expected_test_ids=expected_test_ids,
            mmlu_data_identity=mmlu_data_identity,
            execution=execution,
            data_identity=prepared.data_identity,
            model_identity=prepared.model_identity,
            source_identity=prepared.source_identity,
            experiment_identity=prepared.experiment_identity,
            best_loader=best_loader,
            device=getattr(prepared.runtime, "device"),
            evaluation_mode="pilot",
        )
    runtime_observation = _mmlu_runtime_observation(
        started_at, device=getattr(prepared.runtime, "device")
    )
    _write_pilot_mmlu_orchestration_metadata(
        result, metadata, runtime_observation=runtime_observation
    )
    return result


def _load_piqa_rows_for_stage(config: PilotConfig, assets: object) -> object:
    """Load PIQA lazily; dependency-injected test assets may provide rows directly."""
    loader = getattr(assets, "load_piqa_rows", None)
    if callable(loader):
        return loader(config)
    if hasattr(assets, "piqa_rows"):
        return getattr(assets, "piqa_rows")
    raise OrchestrationError(
        "assets loader must provide a lazy PIQA rows loader for the PIQA stage"
    )


def _require_test_dependencies(
    nonformal_test_mode: bool, *dependencies: object | None
) -> None:
    if type(nonformal_test_mode) is not bool:
        raise TypeError("nonformal_test_mode must be a boolean")
    if any((value is not None for value in dependencies)) and (not nonformal_test_mode):
        raise OrchestrationError(
            "injected evaluation dependencies require nonformal_test_mode=True"
        )
