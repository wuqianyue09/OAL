"""Staged pilot smoke workflow sequencing."""

from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from ..assets import validate_data_manifest
from ..config import PilotConfig
from ..paths import atomic_write_json, collect_preflight
from ..workflows.config import apply_pilot_overrides
from ..workflows.errors import OrchestrationError, _exception_record
from ..workflows.prepare import PreparedPilotRun, prepare_pilot_run
from ..runtime_execution import (
    _method_requires_oal_attention,
    _runtime_attention_execution,
)
from ..smoke_runtime import (
    _failed_smoke_stage,
    _persist_smoke_failure,
    _smoke_evidence_kind,
    _write_smoke_stage,
    build_production_smoke_runtime,
    run_production_smoke_stage,
)

SMOKE_REPORT_FILENAME = "smoke.json"
SMOKE_STAGE_DIRECTORY = "smoke_stages"


def run_smoke(
    config: PilotConfig,
    *,
    run_dir: str | Path,
    preflight_only: bool,
    runtime_builder: Callable[[PilotConfig, PreparedPilotRun], object] | None = None,
    stage_runner: (
        Callable[[str, object, PilotConfig, PreparedPilotRun], Mapping[str, object]]
        | None
    ) = None,
    manifest_validator: Callable[..., dict[str, object]] = validate_data_manifest,
    preflight_collector: Callable[..., Mapping[str, object]] = collect_preflight,
    source_paths: Sequence[str | Path] | None = None,
    hd_manifest_path: str | Path | None = None,
    hd_probe_path: str | Path | None = None,
) -> dict[str, object]:
    """Run production-bound stages and persist evidence for each one.

    Normal calls bind the model/asset runtime and the concrete stage runner in
    this module.  Tests may still inject small fakes, but a CLI invocation no
    longer has a path that can report a successful normal smoke without
    exercising the production implementation.  A smoke run is never a pilot
    completion, irrespective of its outcome.
    """
    prepared = prepare_pilot_run(
        config,
        run_dir=run_dir,
        require_cuda=True,
        require_bf16=True,
        require_oal_attention=_method_requires_oal_attention(config.method),
        manifest_validator=manifest_validator,
        preflight_collector=preflight_collector,
        source_paths=source_paths,
    )
    stages: list[dict[str, object]] = [
        {"name": "preflight", "state": "passed", "details": prepared.preflight},
        {
            "name": "data_manifest",
            "state": "passed",
            "sequence_length": prepared.config.sequence_length,
        },
    ]
    report: dict[str, object] = {
        "schema_version": 1,
        "kind": _smoke_evidence_kind(prepared.config, lora_kind="qwen_lora_smoke"),
        "method": prepared.config.method,
        "run_dir": str(prepared.run_dir),
        "pilot_completed": False,
        "stages": stages,
    }
    report_path = prepared.run_dir / SMOKE_REPORT_FILENAME
    stage_directory = prepared.run_dir / SMOKE_STAGE_DIRECTORY
    try:
        stage_directory.mkdir(exist_ok=False)
        for stage in stages:
            _write_smoke_stage(stage_directory, prepared, stage)
        if preflight_only:
            report["outcome"] = "preflight_complete"
            atomic_write_json(report_path, report)
            return report
    except BaseException as exc:
        _persist_smoke_failure(report_path, report, prepared.status_store, exc)
        raise
    bound_builder = runtime_builder or partial(
        build_production_smoke_runtime,
        hd_manifest_path=hd_manifest_path,
        hd_probe_path=hd_probe_path,
    )
    bound_stage_runner = stage_runner or run_production_smoke_stage
    try:
        runtime = bound_builder(prepared.config, prepared)
        try:
            runtime_bundle = getattr(runtime, "bundle", None)
            identity_source = (
                runtime_bundle
                if hasattr(runtime_bundle, "attention_execution")
                else runtime
            )
            attention_execution = _runtime_attention_execution(
                prepared.config, identity_source
            )
            runtime_identity = {
                "state": "passed",
                "attention_execution": attention_execution,
            }
            report["runtime_identity"] = runtime_identity
            identity_stage = {"name": "runtime_identity", **runtime_identity}
            _write_smoke_stage(stage_directory, prepared, identity_stage)
            stages.append(identity_stage)
        except BaseException as exc:
            identity_stage = {**_failed_smoke_stage("runtime_identity", exc)}
            report["runtime_identity"] = {
                key: value for (key, value) in identity_stage.items() if key != "name"
            }
            _write_smoke_stage(stage_directory, prepared, identity_stage)
            stages.append(identity_stage)
            raise
        for stage_name in ("short_forward", "operator_alignment", "full_model_step"):
            stage_details: Mapping[str, object] | None = None
            try:
                stage_result = bound_stage_runner(
                    stage_name, runtime, prepared.config, prepared
                )
                if not isinstance(stage_result, Mapping):
                    raise OrchestrationError(
                        f"smoke stage {stage_name} must return a mapping"
                    )
                stage_details = stage_result
                if stage_details.get("state") != "passed":
                    if "state" not in stage_details:
                        message = f"smoke stage {stage_name} must explicitly state state='passed'"
                    else:
                        message = f"smoke stage {stage_name} did not pass: {stage_details['state']!r}"
                    raise OrchestrationError(message)
                stage = {**dict(stage_details), "name": stage_name}
                _write_smoke_stage(stage_directory, prepared, stage)
                stages.append(stage)
            except BaseException as exc:
                failed_stage = {**_failed_smoke_stage(stage_name, exc)}
                if stage_details is not None:
                    failed_stage["reported_details"] = dict(stage_details)
                _write_smoke_stage(stage_directory, prepared, failed_stage)
                stages.append(failed_stage)
                raise
        report["outcome"] = "smoke_complete"
        atomic_write_json(report_path, report)
        return report
    except BaseException as exc:
        _persist_smoke_failure(report_path, report, prepared.status_store, exc)
        raise
