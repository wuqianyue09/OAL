"""Shared provenance checks for loading completed optional-evaluation sidecars."""

from __future__ import annotations
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path


def require_reusable_evaluation_summary(
    summary: Mapping[str, object],
    *,
    task: str,
    schema: str,
    run_dir: Path,
    method: str,
    execution: Mapping[str, object],
    best_checkpoint: Mapping[str, object],
    checkpoint_context: Mapping[str, object],
    bundle: Mapping[str, object],
    protocol: str,
    predictions_path: Path,
    identity_path: Path,
) -> dict[str, object]:
    """Require exact run/checkpoint/data provenance and return the task metrics."""
    expected = {
        "schema": schema,
        "kind": f"qwen_lora_{task}_evaluation",
        "status": "completed",
        "evaluation_scope": "pilot_single_seed",
        "read_only_evaluation": True,
        "run_dir": str(run_dir.resolve()),
        "method": method,
        "execution": dict(execution),
        "best_checkpoint": dict(best_checkpoint),
        "checkpoint_context": dict(checkpoint_context),
        "bundle": dict(bundle),
        "protocol": protocol,
        "failed_count": 0,
        "predictions": {
            "path": str(predictions_path.absolute()),
            "identity_path": str(identity_path.absolute()),
        },
    }
    for field, expected_value in expected.items():
        if summary.get(field) != expected_value:
            raise ValueError(
                f"existing {task} sidecar summary field {field!r} conflicts with this request"
            )
    started = _parse_utc_timestamp(
        summary.get("started_at"), task=task, field="started_at"
    )
    completed = _parse_utc_timestamp(
        summary.get("completed_at"), task=task, field="completed_at"
    )
    if completed < started:
        raise ValueError(f"existing {task} sidecar summary timestamps are invalid")
    metrics = summary.get(task)
    if not isinstance(metrics, Mapping):
        raise ValueError(f"existing {task} sidecar summary lacks task metrics")
    return dict(metrics)


def _parse_utc_timestamp(value: object, *, task: str, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"existing {task} sidecar summary {field} is invalid")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"existing {task} sidecar summary {field} is invalid") from exc
    if parsed.tzinfo != timezone.utc:
        raise ValueError(f"existing {task} sidecar summary {field} is invalid")
    return parsed
