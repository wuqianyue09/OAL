from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from ..config import MethodName, PilotConfig, from_json_file


def apply_pilot_overrides(
    config: PilotConfig,
    *,
    method: MethodName | str | None = None,
    tuning_mode: str | None = None,
    sequence_length: int | None = None,
    runs_root: str | Path | None = None,
) -> PilotConfig:
    """Rebuild then validate an override set while preserving profile and scope."""
    if not isinstance(config, PilotConfig):
        raise TypeError("config must be a PilotConfig")
    changes: dict[str, object] = {}
    if method is not None:
        changes["method"] = method
    if tuning_mode is not None:
        changes["tuning_mode"] = tuning_mode
    if sequence_length is not None:
        changes["sequence_length"] = sequence_length
    if runs_root is not None:
        changes["runs_root"] = Path(runs_root)
    candidate = replace(config, **changes)
    candidate.validate()
    return candidate


def load_pilot_config(
    config_path: str | Path,
    *,
    method: MethodName | str | None = None,
    tuning_mode: str | None = None,
    sequence_length: int | None = None,
    runs_root: str | Path | None = None,
) -> PilotConfig:
    """Load a JSON config, then validate the complete overridden profile."""
    return apply_pilot_overrides(
        from_json_file(config_path),
        method=method,
        tuning_mode=tuning_mode,
        sequence_length=sequence_length,
        runs_root=runs_root,
    )
