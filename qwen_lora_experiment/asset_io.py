"""Shared JSON validation and immutable asset file publication."""

from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
from .paths import canonical_json, sha256_file
import json
import os
import tempfile


def _require_nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_json_object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    try:
        canonical_json(dict(value))
    except ValueError as exc:
        raise ValueError(f"{name} must be JSON-serializable: {exc}") from exc
    return dict(value)


def _validate_sha256(digest: object, name: str) -> str:
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any((character not in "0123456789abcdef" for character in digest))
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _load_json_asset(
    path: str | Path, *, expected_sha256: str | None, asset_name: str
) -> Mapping[str, object]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"{asset_name} asset does not exist: {source}")
    if expected_sha256 is not None:
        _validate_sha256(expected_sha256, "expected_sha256")
        if sha256_file(source) != expected_sha256:
            raise ValueError(f"{asset_name} asset SHA-256 mismatch: {source}")
    try:
        with source.open(encoding="utf-8") as asset_file:
            raw = json.load(asset_file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{asset_name} asset JSON is invalid: {exc.msg}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"{asset_name} asset must be a JSON object")
    return raw


def _require_exact_keys(
    value: Mapping[str, object],
    allowed: set[str],
    path: str,
    *,
    required: set[str] | None = None,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{path} has unknown field(s): {', '.join(unknown)}")
    required_fields = allowed if required is None else required
    missing = sorted(required_fields - set(value))
    if missing:
        raise ValueError(f"{path} is missing required field(s): {', '.join(missing)}")


def _write_json_no_clobber(destination: Path, value: Mapping[str, object]) -> None:
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"asset parent directory does not exist: {destination.parent}"
        )
    encoded = (canonical_json(value) + "\n").encode("utf-8")
    temporary_path = _write_temporary_bytes(destination, encoded)
    try:
        try:
            os.link(temporary_path, destination)
        except FileExistsError:
            raise FileExistsError(
                f"immutable asset already exists: {destination}"
            ) from None
        _fsync_directory(destination.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_temporary_bytes(destination: Path, content: bytes) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        return temporary_path
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
