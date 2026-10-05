"""Shared read-only loading for optional completed-run evaluations."""

from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from .. import runtime_identity
from ..config import PilotConfig
from ..experiment_contract import BEST_ADAPTER_FILENAME
from ..paths import sha256_file
from .errors import OrchestrationError

OPTIONAL_BENCHMARK_MAX_SEQUENCE_LENGTH = 4096


def utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True)
class CompletedRun:
    """Prepared model plus validated evidence for one immutable adapter."""

    run_dir: Path
    prepared: object
    best_checkpoint: dict[str, object]


def evaluation_identity(
    config: PilotConfig,
    completed: CompletedRun,
    bundle: Mapping[str, object],
    execution: Mapping[str, object],
    *,
    task: str,
    protocol: str,
    checkpoint_context: dict[str, object],
) -> dict[str, object]:
    """Build the shared identity recorded by optional benchmark workflows."""
    prepared = completed.prepared
    return {
        "schema": f"qwen_lora_{task}_predictions_identity_v1",
        "task": task,
        "read_only_evaluation": True,
        "method": config.method,
        "sequence_limit": OPTIONAL_BENCHMARK_MAX_SEQUENCE_LENGTH,
        "protocol": protocol,
        "best_checkpoint": completed.best_checkpoint,
        "checkpoint_context": checkpoint_context,
        "execution": dict(execution),
        "data_identity": getattr(prepared, "data_identity"),
        "model_identity": getattr(prepared, "model_identity"),
        "source_identity": getattr(prepared, "source_identity"),
        "experiment_identity": getattr(prepared, "experiment_identity"),
        "bundle": dict(bundle),
    }


def evaluation_summary(
    completed: CompletedRun,
    identity: Mapping[str, object],
    *,
    started_at: str,
    completed_at: str,
    predictions_path: Path,
    identity_path: Path,
    metrics: Mapping[str, object],
) -> dict[str, object]:
    """Build a completed summary using the identity already used for reuse."""
    task = identity["task"]
    return {
        "schema": f"qwen_lora_pilot_{task}_evaluation_v1",
        "kind": f"qwen_lora_{task}_evaluation",
        "status": "completed",
        "evaluation_scope": "pilot_single_seed",
        "read_only_evaluation": True,
        "started_at": started_at,
        "completed_at": completed_at,
        "run_dir": str(completed.run_dir.resolve()),
        "method": identity["method"],
        "execution": identity["execution"],
        "best_checkpoint": identity["best_checkpoint"],
        "checkpoint_context": identity["checkpoint_context"],
        "bundle": identity["bundle"],
        "protocol": identity["protocol"],
        "failed_count": 0,
        "predictions": {
            "path": str(predictions_path.absolute()),
            "identity_path": str(identity_path.absolute()),
        },
        task: metrics,
    }


def prepare_completed_run(
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
    manifest_validator: Callable[..., dict[str, object]] | None = None,
    source_paths: Sequence[str | Path] | None = None,
    device: object | None = None,
    hd_manifest_path: str | Path | None = None,
    hd_probe_path: str | Path | None = None,
    allow_source_drift: bool = False,
    nonformal_test_mode: bool = False,
) -> CompletedRun:
    """Rebuild a run in read-only mode and load its own best adapter."""
    if evaluation_mode != "pilot":
        raise ValueError("optional benchmark evaluation is pilot-only")
    if not isinstance(config, PilotConfig):
        raise TypeError("config must be a PilotConfig")
    config.validate()
    if type(nonformal_test_mode) is not bool:
        raise TypeError("nonformal_test_mode must be a boolean")
    injected = any(
        (
            value is not None
            for value in (
                assets_loader,
                model_builder,
                model_identity_collector,
                prepare_runner,
                best_loader,
                manifest_validator,
            )
        )
    )
    if injected and (not nonformal_test_mode):
        raise ValueError(
            "injected evaluation dependencies require nonformal_test_mode=True"
        )
    destination = Path(run_dir)
    _require_completed_status(destination)
    if prepare_runner is None:
        from .evaluation_runtime import _prepare_offline_evaluation

        prepare_runner = _prepare_offline_evaluation
    if best_loader is None:
        from ..checkpointing import load_best_adapter

        best_loader = load_best_adapter
    prepare_kwargs: dict[str, object] = {
        "run_dir": destination,
        "assets_loader": assets_loader,
        "model_builder": model_builder,
        "model_identity_collector": model_identity_collector,
        "source_paths": source_paths,
        "device": device,
        "allow_source_drift": allow_source_drift,
        "nonformal_test_mode": nonformal_test_mode,
        "evaluation_mode": "pilot",
        "read_only_sidecar": True,
    }
    if hd_manifest_path is not None:
        prepare_kwargs["hd_manifest_path"] = hd_manifest_path
    if hd_probe_path is not None:
        prepare_kwargs["hd_probe_path"] = hd_probe_path
    if manifest_validator is not None:
        prepare_kwargs["manifest_validator"] = manifest_validator
    prepared = prepare_runner(config, **prepare_kwargs)
    prepared_run_dir = Path(getattr(prepared, "run_dir"))
    if prepared_run_dir.resolve() != destination.resolve():
        raise OrchestrationError(
            "prepared evaluation runtime belongs to a different run directory"
        )
    runtime = getattr(prepared, "runtime")
    context = getattr(prepared, "checkpoint_context")
    if getattr(context, "method", None) != config.method:
        raise OrchestrationError(
            "checkpoint context method does not match effective configuration"
        )
    loaded = best_loader(
        destination,
        model=getattr(runtime, "model"),
        kernel_bank=getattr(runtime, "kernel_bank"),
        context=context,
    )
    evidence = _checkpoint_record(loaded, destination=destination, method=config.method)
    return CompletedRun(
        run_dir=destination, prepared=prepared, best_checkpoint=evidence
    )


def reference_session(
    config: PilotConfig,
    prepared: object,
    *,
    session_factory: Callable[..., object] | None = None,
) -> object:
    """Return the method-specific quality session without unifying task logic."""
    if not config.uses_grouped_quadratic_base:
        return nullcontext()
    if session_factory is None:
        from .evaluation_reference import _grouped_quadratic_reference_session

        session_factory = _grouped_quadratic_reference_session
    runtime = getattr(prepared, "runtime")
    return session_factory(
        getattr(runtime, "model"), expected_layer_ids=config.replacement_layer_ids
    )


def evaluation_execution(
    config: PilotConfig,
    prepared: object,
    *,
    nonformal_test_mode: bool,
    grouped_alignment_runner: Callable[..., Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Describe the actual scorer path and attach grouped reference alignment."""
    source = getattr(prepared, "execution")
    if not isinstance(source, Mapping):
        raise OrchestrationError("prepared evaluation execution must be a mapping")
    execution = dict(source)
    execution.update(model_scope_execution(config))
    if not config.uses_grouped_quadratic_base:
        return execution
    attention = execution.get("attention")
    if not isinstance(attention, Mapping) or attention.get("method") != config.method:
        raise OrchestrationError(
            "grouped evaluation execution lacks its attention identity"
        )
    if grouped_alignment_runner is None:
        if nonformal_test_mode:
            alignment: Mapping[str, object] = {
                "schema": "grouped_reference_alignment_set_v1",
                "status": "not_run_nonformal_test",
                "layers": [],
            }
        else:
            alignment = _run_grouped_reference_alignment(
                method=config.method,
                kernel_bank=getattr(getattr(prepared, "runtime"), "kernel_bank"),
                layer_ids=config.replacement_layer_ids,
                device=getattr(getattr(prepared, "runtime"), "device"),
                seed=config.seed_derivations.diagnostic_inputs_seed,
            )
    else:
        alignment = grouped_alignment_runner(
            kernel_bank=getattr(getattr(prepared, "runtime"), "kernel_bank"),
            layer_ids=config.replacement_layer_ids,
            device=getattr(getattr(prepared, "runtime"), "device"),
            seed=config.seed_derivations.diagnostic_inputs_seed,
        )
    normalized_alignment = _mapping(alignment, "grouped reference alignment")
    training_attention = dict(attention)
    execution["attention"] = {
        **training_attention,
        "training_execution": training_attention.get("execution"),
        "execution": "reference",
        "public_callable": "oal_attention.oal_attention",
        "scope": "quality_only_no_triton_performance_claim",
        "max_sequence_length": OPTIONAL_BENCHMARK_MAX_SEQUENCE_LENGTH,
    }
    execution["grouped_reference_alignment"] = normalized_alignment
    return execution


def model_scope_execution(config: PilotConfig) -> dict[str, object]:
    """Describe the model and requested/actual attention protocol in results."""
    geometry = config.geometry
    return {
        "model_profile": config.resolved_model_profile,
        "model_geometry": {
            "num_layers": geometry.num_layers,
            "hidden_size": geometry.hidden_size,
            "num_query_heads": geometry.num_query_heads,
            "num_kv_heads": geometry.num_kv_heads,
            "head_dim": geometry.head_dim,
        },
        "requested_replacement_layer_ids_zero_based": list(config.replacement_layers),
        "actual_replacement_layer_ids_zero_based": list(config.replacement_layer_ids),
    }


def checkpoint_context_record(context: object) -> dict[str, object]:
    """Serialize the three stable checkpoint identity mappings."""
    return {
        "config_identity": _mapping(
            getattr(context, "config_identity"), "config_identity"
        ),
        "data_identity": _mapping(getattr(context, "data_identity"), "data_identity"),
        "model_identity": _mapping(
            getattr(context, "model_identity"), "model_identity"
        ),
    }


def bundle_identity(path: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    """Keep the full immutable manifest beside its resolved local path."""
    value = _mapping(manifest, "benchmark bundle manifest")
    if not isinstance(value.get("protocol"), str) or not value["protocol"]:
        raise ValueError("benchmark bundle manifest requires a protocol")
    digest = value.get("bundle_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("benchmark bundle manifest requires bundle_sha256")
    if not isinstance(value.get("tokenizer"), Mapping):
        raise ValueError("benchmark bundle manifest requires tokenizer identity")
    return {"path": str(path.absolute()), "manifest": value}


def require_runtime_tokenizer_identity(
    config: PilotConfig,
    prepared: object,
    manifest: Mapping[str, object],
    *,
    nonformal_test_mode: bool,
) -> None:
    """Bind a production benchmark bundle to the tokenizer in the rebuilt run."""
    if nonformal_test_mode:
        return
    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise OrchestrationError("benchmark bundle is missing tokenizer identity")
    runtime_identity._assert_runtime_tokenizer_matches_manifest(
        config, getattr(prepared, "runtime"), {"tokenizer": dict(tokenizer)}
    )


def _checkpoint_record(
    loaded: object, *, destination: Path, method: str
) -> dict[str, object]:
    path = Path(getattr(loaded, "path"))
    expected_path = destination / BEST_ADAPTER_FILENAME
    if path.resolve() != expected_path.resolve() or not expected_path.is_file():
        raise OrchestrationError("loaded checkpoint is not this run's best_adapter.pt")
    step = getattr(loaded, "step")
    validation_nll = getattr(loaded, "validation_nll")
    if type(step) is not int or step < 0:
        raise OrchestrationError("loaded checkpoint step is invalid")
    if not isinstance(validation_nll, (int, float)) or not math.isfinite(
        float(validation_nll)
    ):
        raise OrchestrationError("loaded checkpoint validation NLL is invalid")
    if getattr(loaded, "method", None) != method:
        raise OrchestrationError(
            "loaded checkpoint method does not match effective configuration"
        )
    return {
        "path": str(expected_path.resolve()),
        "sha256": sha256_file(expected_path),
        "step": step,
        "validation_nll": float(validation_nll),
        "method": method,
    }


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return dict(value)


def _require_completed_status(run_dir: Path) -> None:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"evaluation run directory does not exist: {run_dir}")
    path = run_dir / "status.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(
            f"completed evaluation run is missing status.json: {run_dir}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"completed evaluation run has invalid status.json: {run_dir}"
        ) from exc
    if not isinstance(value, Mapping) or value.get("state") not in {
        "trained",
        "completed",
    }:
        raise ValueError(
            "optional benchmark evaluation requires status.json state 'trained' or 'completed'"
        )


def _run_grouped_reference_alignment(
    *,
    method: str,
    kernel_bank: object,
    layer_ids: Sequence[int],
    device: object,
    seed: int,
) -> dict[str, object]:
    if method == "grouped_quadratic":
        from ..attention.grouped import grouped_quadratic_reference_alignment as align

        schema = "grouped_reference_alignment_set_v1"
    else:
        raise ValueError("grouped reference alignment requires a grouped-base method")
    return {
        "schema": schema,
        "sequence_length": 128,
        "seed": seed,
        "layers": [
            align(
                parameter_bank=kernel_bank, layer_id=layer_id, device=device, seed=seed
            )
            for layer_id in layer_ids
        ],
    }


def _require_evaluation_mode(value: object) -> str:
    if value != "pilot":
        raise ValueError("evaluation_mode must be 'pilot'")
    return "pilot"
