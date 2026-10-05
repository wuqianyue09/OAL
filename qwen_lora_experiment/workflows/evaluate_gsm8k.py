"""Read-only GSM8K evaluation for one completed run."""

from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
from ..config import PilotConfig
from ..evaluators.artifacts import (
    load_reusable_sidecar,
    publish_sidecar,
    sidecar_artifact_paths,
    sidecar_evaluation_session,
)
from ..evaluators.gsm8k import Gsm8kScore, evaluate_gsm8k, reusable_gsm8k_score
from ..evaluators.reuse import require_reusable_evaluation_summary
from ..gsm8k import (
    GSM8K_PROTOCOL,
    format_gsm8k_prompt,
    gsm8k_bundle_path,
    load_gsm8k_bundle,
)
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
class Gsm8kEvaluationResult:
    final_path: Path
    predictions_path: Path
    identity_path: Path
    best_checkpoint: dict[str, object]
    gsm8k: Gsm8kScore


def execute_pilot_gsm8k_evaluation(
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
    evaluation_runner: Callable[..., Gsm8kScore] | None = None,
    reference_session_factory: Callable[..., object] | None = None,
    grouped_alignment_runner: Callable[..., Mapping[str, object]] | None = None,
    manifest_validator: Callable[..., dict[str, object]] | None = None,
    source_paths: Sequence[str | Path] | None = None,
    device: object | None = None,
    hd_manifest_path: str | Path | None = None,
    hd_probe_path: str | Path | None = None,
    allow_source_drift: bool = False,
    nonformal_test_mode: bool = False,
) -> Gsm8kEvaluationResult:
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
        bundle_loader = load_gsm8k_bundle
    if evaluation_runner is None:
        evaluation_runner = evaluate_gsm8k
    bundle_path = gsm8k_bundle_path(config.data_root)
    assignments, test_rows, manifest = bundle_loader(bundle_path)
    if not isinstance(manifest, Mapping):
        raise TypeError("GSM8K bundle loader must return a manifest mapping")
    require_runtime_tokenizer_identity(
        config, completed.prepared, manifest, nonformal_test_mode=nonformal_test_mode
    )
    expected_ids = tuple((row["id"] for row in test_rows))
    expected_records = _expected_gsm8k_records(assignments, test_rows)
    prepared = completed.prepared
    runtime = getattr(prepared, "runtime")
    bundle = bundle_identity(bundle_path, manifest)
    checkpoint_context = checkpoint_context_record(
        getattr(prepared, "checkpoint_context")
    )
    _, predictions_path, identity_path = sidecar_artifact_paths(
        completed.run_dir, task="gsm8k"
    )
    with sidecar_evaluation_session(completed.run_dir, task="gsm8k"):
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
                task="gsm8k",
                protocol=GSM8K_PROTOCOL,
                checkpoint_context=checkpoint_context,
            )
            reusable = load_reusable_sidecar(
                completed.run_dir, task="gsm8k", identity=identity
            )
            if reusable is not None:
                metrics = require_reusable_evaluation_summary(
                    reusable.summary,
                    task="gsm8k",
                    schema="qwen_lora_pilot_gsm8k_evaluation_v1",
                    run_dir=completed.run_dir,
                    method=config.method,
                    execution=execution,
                    best_checkpoint=completed.best_checkpoint,
                    checkpoint_context=checkpoint_context,
                    bundle=bundle,
                    protocol=GSM8K_PROTOCOL,
                    predictions_path=reusable.predictions_path,
                    identity_path=reusable.identity_path,
                )
                score = reusable_gsm8k_score(
                    predictions=reusable.predictions,
                    metrics=metrics,
                    expected_records=expected_records,
                )
                return Gsm8kEvaluationResult(
                    final_path=reusable.final_path,
                    predictions_path=reusable.predictions_path,
                    identity_path=reusable.identity_path,
                    best_checkpoint=completed.best_checkpoint,
                    gsm8k=score,
                )
            started_at = utc_timestamp()
            score = evaluation_runner(
                model=getattr(runtime, "model"),
                tokenizer=getattr(runtime, "tokenizer"),
                test_rows=test_rows,
                assignments=assignments,
                expected_ids=expected_ids,
                sequence_limit=OPTIONAL_BENCHMARK_MAX_SEQUENCE_LENGTH,
                kernel_bank=getattr(runtime, "kernel_bank"),
                device=getattr(runtime, "device"),
            )
            completed_at = utc_timestamp()
        if not isinstance(score, Gsm8kScore):
            raise TypeError("GSM8K evaluator must return Gsm8kScore")
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
            task="gsm8k",
            summary=summary,
            predictions=score.predictions,
            identity=identity,
        )
    return Gsm8kEvaluationResult(
        final_path=published.final_path,
        predictions_path=published.predictions_path,
        identity_path=published.identity_path,
        best_checkpoint=completed.best_checkpoint,
        gsm8k=score,
    )


def _require_mode_and_injections(
    evaluation_mode: str,
    nonformal_test_mode: bool,
    dependencies: Sequence[object | None],
) -> None:
    if evaluation_mode != "pilot":
        raise ValueError("GSM8K evaluation is pilot-only")
    if type(nonformal_test_mode) is not bool:
        raise TypeError("nonformal_test_mode must be a boolean")
    if any((value is not None for value in dependencies)) and (not nonformal_test_mode):
        raise ValueError(
            "injected evaluation dependencies require nonformal_test_mode=True"
        )


def _expected_gsm8k_records(
    assignments: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, str]]:
    if len(assignments) != len(test_rows):
        raise ValueError("GSM8K bundle assignments do not cover every test row")
    records: dict[str, dict[str, str]] = {}
    for index, (assignment, row) in enumerate(zip(assignments, test_rows, strict=True)):
        identifier = row.get("id")
        gold = row.get("gold_answer")
        demonstrations = assignment.get("demonstrations")
        if (
            not isinstance(identifier, str)
            or not identifier
            or (not isinstance(gold, str))
            or (not gold)
            or (assignment.get("id") != identifier)
            or (not isinstance(demonstrations, list))
        ):
            raise ValueError(f"GSM8K bundle row {index} or assignment is invalid")
        prompt = format_gsm8k_prompt(demonstrations, row)
        records[identifier] = {
            "gold_answer": gold,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        }
    if len(records) != len(test_rows):
        raise ValueError("GSM8K bundle test IDs must be unique")
    return records
