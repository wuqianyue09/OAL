"""Import-safe smoke construction, stage dispatch, and evidence persistence."""

from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
from .config import PilotConfig
from .paths import atomic_write_json
from .telemetry import RunStateSink
from .workflows.errors import OrchestrationError, _exception_record
from .workflows.prepare import PreparedPilotRun, _record_preparation_failure
from . import runtime_execution
from .smoke_alignment import _production_operator_alignment
from .smoke_lora import _production_lora_full_model_step
from .smoke_common import ProductionSmokeRuntime, _smoke_input_ids


def build_production_smoke_runtime(
    config: PilotConfig,
    prepared: PreparedPilotRun,
    *,
    hd_manifest_path: str | Path | None = None,
    hd_probe_path: str | Path | None = None,
) -> ProductionSmokeRuntime:
    """Load immutable assets and the real Qwen bundle after preflight succeeds."""
    if prepared.config != config:
        raise OrchestrationError(
            "prepared smoke configuration does not match runtime configuration"
        )
    from .assets import TrainingAssets
    from .model_setup import build_model_bundle

    device = runtime_execution._production_cuda_device()
    assets = TrainingAssets.load(config)
    bundle = build_model_bundle(
        config,
        device=device,
        hd_manifest_path=hd_manifest_path,
        hd_probe_path=hd_probe_path,
    )
    required_fields = ("model", "kernel_bank", "optimizer", "scheduler", "device")
    if any((not hasattr(bundle, field) for field in required_fields)):
        raise OrchestrationError(
            "production smoke bundle is missing a required runtime field"
        )
    return ProductionSmokeRuntime(bundle=bundle, assets=assets)


def run_production_smoke_stage(
    stage_name: str, runtime: object, config: PilotConfig, prepared: PreparedPilotRun
) -> Mapping[str, object]:
    """Execute one real smoke stage; there is intentionally no fake fallback."""
    if not isinstance(runtime, ProductionSmokeRuntime):
        raise OrchestrationError(
            "production smoke stages require ProductionSmokeRuntime"
        )
    if prepared.config != config:
        raise OrchestrationError(
            "prepared smoke configuration does not match stage configuration"
        )
    if stage_name == "short_forward":
        return _production_short_forward(runtime, config)
    if stage_name == "operator_alignment":
        return _production_operator_alignment(runtime, config)
    if stage_name == "full_model_step":
        return _production_full_model_step(runtime, config)
    raise ValueError(f"unknown production smoke stage: {stage_name!r}")


def _production_short_forward(
    runtime: ProductionSmokeRuntime, config: PilotConfig
) -> dict[str, object]:
    """Run the real model once at the approved short diagnostic length."""
    import torch

    bundle = runtime.bundle
    model = getattr(bundle, "model")
    kernel_bank = getattr(bundle, "kernel_bank")
    input_ids = _smoke_input_ids(
        assets=runtime.assets,
        sequence_length=config.diagnostic_sequence_length,
        device=getattr(bundle, "device"),
    )
    model.eval()
    kernel_bank.eval()
    with torch.no_grad():
        logits = runtime_execution.model_logits(model, input_ids)
    runtime_execution.require_finite_tensor(logits, "short forward logits")
    return {
        "state": "passed",
        "sequence_length": config.diagnostic_sequence_length,
        "input_shape": list(input_ids.shape),
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
    }


def _production_full_model_step(
    runtime: ProductionSmokeRuntime, config: PilotConfig
) -> dict[str, object]:
    """Dispatch the full-length smoke gate without changing legacy LoRA evidence."""
    if config.tuning_mode == "lora":
        return _production_lora_full_model_step(runtime, config)
    raise OrchestrationError(f"unsupported smoke tuning mode: {config.tuning_mode!r}")


def _write_smoke_stage(
    directory: Path, prepared: PreparedPilotRun, stage: Mapping[str, object]
) -> None:
    """Persist one stage separately so aggregate-report loss cannot erase proof."""
    name = stage.get("name")
    state = stage.get("state")
    if not isinstance(name, str) or not name:
        raise OrchestrationError("smoke stage evidence requires a non-empty name")
    if state not in {"passed", "failed"}:
        raise OrchestrationError(
            "smoke stage evidence requires state 'passed' or 'failed'"
        )
    atomic_write_json(
        directory / f"{name}.json",
        {
            "schema_version": 1,
            "kind": _smoke_evidence_kind(
                prepared.config, lora_kind="qwen_lora_smoke_stage"
            ),
            "method": prepared.config.method,
            "run_dir": str(prepared.run_dir),
            **dict(stage),
        },
    )


def _failed_smoke_stage(name: str, error: BaseException) -> dict[str, object]:
    """Represent a failed stage independently of the aggregate smoke report."""
    return {"name": name, "state": "failed", "failure": _exception_record(error)}


def _persist_smoke_failure(
    report_path: Path,
    report: dict[str, object],
    status_store: RunStateSink,
    error: BaseException,
) -> None:
    """Best-effort terminal evidence that never hides the original failure."""
    report.update({"outcome": "failed", "failure": _exception_record(error)})
    try:
        atomic_write_json(report_path, report)
    except BaseException:
        pass
    _record_preparation_failure(status_store, error)


def _smoke_evidence_kind(config: PilotConfig, *, lora_kind: str) -> str:
    """Keep Qwen history stable while naming Llama smoke evidence factually."""
    if config.resolved_model_profile == "llama3_2_1b_base":
        if not lora_kind.startswith("qwen_"):
            raise ValueError(
                "Llama smoke evidence kind must derive from a Qwen legacy kind"
            )
        return f"llama_{lora_kind.removeprefix('qwen_')}"
    return lora_kind
