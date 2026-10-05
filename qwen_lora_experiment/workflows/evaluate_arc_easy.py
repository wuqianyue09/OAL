"""Read-only ARC-E evaluation for one completed run."""

from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from ..arc_easy import ARC_EASY_PROTOCOL, arc_easy_bundle_path, load_arc_easy_bundle
from ..config import PilotConfig
from ..evaluators.arc_easy import (
    ArcEasyScore,
    evaluate_arc_easy,
    reusable_arc_easy_score,
)
from ..evaluators.artifacts import (
    load_reusable_sidecar,
    publish_sidecar,
    sidecar_artifact_paths,
    sidecar_evaluation_session,
)
from ..evaluators.reuse import require_reusable_evaluation_summary
from .evaluate_common import (
    OPTIONAL_BENCHMARK_MAX_SEQUENCE_LENGTH,
    bundle_identity,
    checkpoint_context_record,
    evaluation_execution,
    evaluation_identity,
    evaluation_summary,
    prepare_completed_run,
    reference_session,
    require_runtime_tokenizer_identity,
    utc_timestamp,
)


@dataclass(frozen=True)
class ArcEasyEvaluationResult:
    final_path: Path
    predictions_path: Path
    identity_path: Path
    best_checkpoint: dict[str, object]
    arc_easy: ArcEasyScore


def execute_pilot_arc_easy_evaluation(
    config: PilotConfig,
    *,
    run_dir: str | Path,
    evaluation_mode: str = "pilot",
    assets_loader: Callable[[PilotConfig], object] | None = None,
    model_builder: Callable[[PilotConfig, object], object] | None = None,
    model_identity_collector: (
        Callable[[str | Path], Mapping[str, object]] | None
    ) = None,
    prepare_runner: Callable[..., object] | None = None,
    best_loader: Callable[..., object] | None = None,
    bundle_loader: Callable[..., object] | None = None,
    evaluation_runner: Callable[..., ArcEasyScore] | None = None,
    reference_session_factory: Callable[..., object] | None = None,
    grouped_alignment_runner: Callable[..., Mapping[str, object]] | None = None,
    manifest_validator: Callable[..., dict[str, object]] | None = None,
    source_paths: Sequence[str | Path] | None = None,
    device: object | None = None,
    hd_manifest_path: str | Path | None = None,
    hd_probe_path: str | Path | None = None,
    allow_source_drift: bool = False,
    nonformal_test_mode: bool = False,
) -> ArcEasyEvaluationResult:
    """Evaluate the immutable local test bundle without run transitions."""
    _require_mode_and_injections(
        evaluation_mode,
        nonformal_test_mode,
        (
            prepare_runner,
            best_loader,
            bundle_loader,
            evaluation_runner,
            reference_session_factory,
            grouped_alignment_runner,
        ),
    )
    completed = prepare_completed_run(
        config,
        run_dir=run_dir,
        evaluation_mode=evaluation_mode,
        assets_loader=assets_loader,
        model_builder=model_builder,
        model_identity_collector=model_identity_collector,
        prepare_runner=prepare_runner,
        best_loader=best_loader,
        manifest_validator=manifest_validator,
        source_paths=source_paths,
        device=device,
        hd_manifest_path=hd_manifest_path,
        hd_probe_path=hd_probe_path,
        allow_source_drift=allow_source_drift,
        nonformal_test_mode=nonformal_test_mode,
    )
    if bundle_loader is None:
        bundle_loader = load_arc_easy_bundle
    if evaluation_runner is None:
        evaluation_runner = evaluate_arc_easy
    bundle_path = arc_easy_bundle_path(config.data_root)
    rows, manifest = bundle_loader(bundle_path)
    if not isinstance(manifest, Mapping):
        raise TypeError("ARC-E bundle loader must return a manifest mapping")
    require_runtime_tokenizer_identity(
        config, completed.prepared, manifest, nonformal_test_mode=nonformal_test_mode
    )
    expected_ids = tuple((row["id"] for row in rows))
    prepared = completed.prepared
    runtime = getattr(prepared, "runtime")
    bundle = bundle_identity(bundle_path, manifest)
    checkpoint_context = checkpoint_context_record(
        getattr(prepared, "checkpoint_context")
    )
    final_path, predictions_path, identity_path = sidecar_artifact_paths(
        completed.run_dir, task="arc_easy"
    )
    with sidecar_evaluation_session(completed.run_dir, task="arc_easy"):
        with reference_session(
            config, prepared, session_factory=reference_session_factory
        ):
            execution = evaluation_execution(
                config,
                prepared,
                nonformal_test_mode=nonformal_test_mode,
                grouped_alignment_runner=grouped_alignment_runner,
            )
            identity = evaluation_identity(
                config,
                completed,
                bundle,
                execution,
                task="arc_easy",
                protocol=ARC_EASY_PROTOCOL,
                checkpoint_context=checkpoint_context,
            )
            reusable = load_reusable_sidecar(
                completed.run_dir, task="arc_easy", identity=identity
            )
            if reusable is not None:
                metrics = require_reusable_evaluation_summary(
                    reusable.summary,
                    task="arc_easy",
                    schema="qwen_lora_pilot_arc_easy_evaluation_v1",
                    run_dir=completed.run_dir,
                    method=config.method,
                    execution=execution,
                    best_checkpoint=completed.best_checkpoint,
                    checkpoint_context=checkpoint_context,
                    bundle=bundle,
                    protocol=ARC_EASY_PROTOCOL,
                    predictions_path=reusable.predictions_path,
                    identity_path=reusable.identity_path,
                )
                score = reusable_arc_easy_score(
                    predictions=reusable.predictions,
                    metrics=metrics,
                    expected_rows=rows,
                )
                return ArcEasyEvaluationResult(
                    final_path=reusable.final_path,
                    predictions_path=reusable.predictions_path,
                    identity_path=reusable.identity_path,
                    best_checkpoint=completed.best_checkpoint,
                    arc_easy=score,
                )
            started_at = utc_timestamp()
            score = evaluation_runner(
                model=getattr(runtime, "model"),
                tokenizer=getattr(runtime, "tokenizer"),
                rows=rows,
                expected_ids=expected_ids,
                sequence_limit=OPTIONAL_BENCHMARK_MAX_SEQUENCE_LENGTH,
                kernel_bank=getattr(runtime, "kernel_bank"),
                device=getattr(runtime, "device"),
            )
            completed_at = utc_timestamp()
        if not isinstance(score, ArcEasyScore):
            raise TypeError("ARC-E evaluator must return ArcEasyScore")
        summary = evaluation_summary(
            completed,
            identity,
            started_at=started_at,
            completed_at=completed_at,
            predictions_path=predictions_path,
            identity_path=identity_path,
            metrics=score.summary(),
        )
        published = publish_sidecar(
            completed.run_dir,
            task="arc_easy",
            summary=summary,
            predictions=score.predictions,
            identity=identity,
        )
    return ArcEasyEvaluationResult(
        final_path=published.final_path,
        predictions_path=published.predictions_path,
        identity_path=published.identity_path,
        best_checkpoint=completed.best_checkpoint,
        arc_easy=score,
    )


def _require_mode_and_injections(
    evaluation_mode: str,
    nonformal_test_mode: bool,
    dependencies: Sequence[object | None],
) -> None:
    if evaluation_mode != "pilot":
        raise ValueError("ARC-E evaluation is pilot-only")
    if type(nonformal_test_mode) is not bool:
        raise TypeError("nonformal_test_mode must be a boolean")
    if any((value is not None for value in dependencies)) and (not nonformal_test_mode):
        raise ValueError(
            "injected evaluation dependencies require nonformal_test_mode=True"
        )
