"""Durable local-file, provenance, and read-only environment helpers.

The pilot is deliberately usable on a development machine that has neither a
model download nor the training stack installed.  Consequently this module
uses only the standard library at import time and imports PyTorch only inside
the two operations that need it.
"""

from __future__ import annotations
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

SCHEMA_VERSION = 1
_HASH_CHUNK_SIZE = 1024 * 1024
_OPTIONAL_MODEL_FILES = (
    "added_tokens.json",
    "merges.txt",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
_WEIGHT_SUFFIXES = (".bin", ".pt", ".pth", ".safetensors")


def utc_timestamp() -> str:
    """Return a timezone-explicit UTC timestamp suitable for JSON records."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all((character in "0123456789abcdef" for character in value))
    )


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 digest of ``data`` as lowercase hexadecimal."""
    if not isinstance(data, bytes):
        raise TypeError("SHA-256 byte input must be bytes")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash one regular file without loading it wholly into memory."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"SHA-256 input file does not exist: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"SHA-256 input path is not a regular file: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as source_file:
        while chunk := source_file.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    """Serialize JSON deterministically for hashes and append-only records."""
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"value cannot be represented as canonical JSON: {exc}"
        ) from exc


def atomic_write_json(path: str | Path, value: object) -> None:
    """Durably replace a JSON file using a same-directory temporary file."""
    encoded = (canonical_json(value) + "\n").encode("utf-8")
    _atomic_write_bytes(Path(path), encoded)


def atomic_create_json(path: str | Path, value: object) -> None:
    """Create one canonical JSON record without replacing an existing target.

    A same-directory temporary file is fsynced before an atomic hard-link
    creation of the destination. Therefore competing writers or any existing
    target fail without replacing the earlier evidence record. If this call's
    link succeeds but its later directory sync fails, it removes only its own
    installed destination before raising.
    """
    destination = Path(path)
    if destination.is_symlink():
        raise FileExistsError("create-once output path must not be a symlink")
    parent = destination.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"output parent directory does not exist: {parent}")
    encoded = (canonical_json(value) + "\n").encode("utf-8")
    reservation: Path | None = None
    temporary: Path | None = None
    link_succeeded = False
    try:
        try:
            if destination.exists():
                raise FileExistsError(
                    f"create-once output already exists: {destination}"
                )
            reservation = _reserve_create_once_path(destination)
            temporary = _temporary_path(destination)
            with temporary.open("wb") as temporary_file:
                temporary_file.write(encoded)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            try:
                os.link(temporary, destination, follow_symlinks=False)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"create-once output already exists: {destination}"
                ) from exc
            link_succeeded = True
            _fsync_parent(parent)
        finally:
            if temporary is not None:
                _remove_if_exists(temporary)
            if reservation is not None:
                _remove_if_exists(reservation)
            _fsync_parent(parent)
    except BaseException:
        if link_succeeded:
            _remove_if_exists(destination)
        raise


def _reserve_create_once_path(destination: Path) -> Path:
    """Reserve a sibling name so cleanup never touches another writer's files."""
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{destination.name}.plan-b-reservation-",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return Path(raw_path)


def _fsync_parent(parent: Path) -> None:
    descriptor = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_torch_save(path: str | Path, value: object) -> None:
    """Durably replace a PyTorch checkpoint without importing torch at module load."""
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError(
            "atomic_torch_save requires the optional 'torch' package"
        ) from exc
    destination = Path(path)
    temporary_path = _temporary_path(destination)
    try:
        with temporary_path.open("wb") as temporary_file:
            torch.save(value, temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        _replace_and_fsync_parent(temporary_path, destination)
    except BaseException:
        _remove_if_exists(temporary_path)
        raise


def source_file_hashes(paths: Iterable[str | Path]) -> dict[str, str]:
    """Return deterministic absolute-path SHA-256 records for source files."""
    resolved_paths = sorted({Path(path).resolve() for path in paths}, key=str)
    hashes: dict[str, str] = {}
    for source_path in resolved_paths:
        hashes[str(source_path)] = sha256_file(source_path)
    return hashes


def hash_source_files(paths: Iterable[str | Path]) -> dict[str, str]:
    """Compatibility name for :func:`source_file_hashes`."""
    return source_file_hashes(paths)


def collect_local_model_identity(model_path: str | Path) -> dict[str, object]:
    """Fingerprint a local Hugging Face model directory without network access.

    ``identity_sha256`` covers only stable local facts.  ``collected_at`` is
    useful provenance but intentionally excluded so a resume check can compare
    identities across processes.
    """
    root = Path(model_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"local model directory does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"local model path is not a directory: {root}")
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"local model is missing required config.json: {root}")
    config = _read_json_object(config_path, "local model config")
    files: dict[str, str] = {"config.json": sha256_file(config_path)}
    missing_optional_files: list[str] = []
    index_weight_names: set[str] = set()
    for relative_path in _OPTIONAL_MODEL_FILES:
        candidate = root / relative_path
        if not candidate.is_file():
            missing_optional_files.append(relative_path)
            continue
        files[relative_path] = sha256_file(candidate)
        if relative_path.endswith(".index.json"):
            index_weight_names.update(_weight_names_from_index(candidate))
    weight_paths = _find_weight_paths(root, index_weight_names)
    weight_files = [
        {"path": str(weight_path.relative_to(root)), "sha256": sha256_file(weight_path)}
        for weight_path in weight_paths
    ]
    stable_identity: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "local_model_identity",
        "model_path": str(root),
        "hub_revision": _find_local_revision(config),
        "files": dict(sorted(files.items())),
        "missing_optional_files": sorted(missing_optional_files),
        "weight_files": weight_files,
    }
    identity = dict(stable_identity)
    identity["identity_sha256"] = sha256_bytes(
        canonical_json(stable_identity).encode("utf-8")
    )
    identity["collected_at"] = utc_timestamp()
    return identity


def semantic_model_identity_sha256(model_identity: Mapping[str, object]) -> str:
    """Hash model facts that remain stable when the local checkout moves.

    The formal asset gate compares this value to a freshly collected local
    model identity.  Absolute paths and collection time are diagnostics, not
    semantic evidence, and the legacy identity digest is path-bound.
    """
    semantic = dict(model_identity)
    semantic.pop("model_path", None)
    semantic.pop("collected_at", None)
    semantic.pop("identity_sha256", None)
    return sha256_bytes(canonical_json(semantic).encode("utf-8"))


def local_model_identity(model_path: str | Path) -> dict[str, object]:
    """Compatibility name for :func:`collect_local_model_identity`."""
    return collect_local_model_identity(model_path)


def collect_preflight(
    *,
    required_packages: Iterable[str] = ("torch", "transformers", "triton"),
    project_paths: Mapping[str, str | Path] | None = None,
    require_cuda: bool = False,
    require_bf16: bool = False,
    require_oal_attention: bool = False,
) -> dict[str, object]:
    """Read installed capabilities without modifying the environment.

    Missing prerequisites are represented in ``missing_dependencies`` rather
    than raised so a caller can persist a failed status record with the exact
    evidence.  No installer, downloader, or model-loading code is invoked.
    """
    required = tuple(dict.fromkeys(required_packages))
    if any((not isinstance(name, str) or not name for name in required)):
        raise ValueError("required_packages must contain non-empty package names")
    package_reports = {
        name: _probe_import(name)
        for name in sorted(set(required) | {"torch", "transformers", "triton"})
    }
    operator_report = _probe_import("oal_attention")
    missing_dependencies = [
        {
            "name": name,
            "reason": (
                "not installed"
                if package_reports[name]["availability"] == "unavailable"
                else "import error"
            ),
        }
        for name in required
        if not package_reports[name]["available"]
    ]
    cuda_report = _collect_cuda_report(package_reports["torch"])
    if require_cuda and (not cuda_report["available"]):
        missing_dependencies.append({"name": "cuda", "reason": "not available"})
    if require_bf16:
        if not cuda_report["available"]:
            missing_dependencies.append(
                {"name": "bf16", "reason": "CUDA is not available"}
            )
        elif not cuda_report["bf16_supported"]:
            missing_dependencies.append(
                {"name": "bf16", "reason": "CUDA device does not support BF16"}
            )
    if require_oal_attention and (not operator_report["available"]):
        missing_dependencies.append(
            {"name": "oal_attention", "reason": _unavailable_reason(operator_report)}
        )
    resolved_project_paths = _resolve_project_paths(project_paths)
    return {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": utc_timestamp(),
        "ok": not missing_dependencies,
        "python": {
            "implementation": sys.implementation.name,
            "version": sys.version,
            "executable": sys.executable,
        },
        "packages": package_reports,
        "cuda": cuda_report,
        "requirements": {
            "cuda": require_cuda,
            "bf16": require_bf16,
            "oal_attention": require_oal_attention,
        },
        "project_paths": resolved_project_paths,
        "oal_attention": operator_report,
        "missing_dependencies": missing_dependencies,
    }


def _unavailable_reason(report: Mapping[str, object]) -> str:
    """Map a structured import probe to the durable missing-dependency reason."""
    return (
        "not installed" if report["availability"] == "unavailable" else "import error"
    )


def _atomic_write_bytes(destination: Path, content: bytes) -> None:
    temporary_path = _temporary_path(destination)
    try:
        with temporary_path.open("wb") as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        _replace_and_fsync_parent(temporary_path, destination)
    except BaseException:
        _remove_if_exists(temporary_path)
        raise


def _temporary_path(destination: Path) -> Path:
    parent = destination.parent
    if not parent.exists():
        raise FileNotFoundError(
            f"atomic-write target directory does not exist: {parent}"
        )
    if not parent.is_dir():
        raise NotADirectoryError(
            f"atomic-write target parent is not a directory: {parent}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    return Path(temporary_name)


def _remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _replace_and_fsync_parent(temporary_path: Path, destination: Path) -> None:
    """Replace ``destination`` then durably record that rename in its parent."""
    temporary_path.replace(destination)
    try:
        fsync_directory(destination.parent)
    except OSError as exc:
        raise OSError(
            f"atomic replacement completed but target directory could not be fsynced: {destination.parent}"
        ) from exc


def fsync_directory(path: str | Path) -> None:
    """Synchronize a directory entry and preserve the target path in failures."""
    directory = Path(path)
    try:
        directory_descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as exc:
        raise OSError(f"target directory could not be fsynced: {directory}") from exc


def _read_json_object(path: Path, description: str) -> dict[str, object]:
    try:
        with path.open(encoding="utf-8") as source_file:
            value = json.load(source_file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{description} is invalid JSON at {path}: {exc.msg}") from exc
    except OSError as exc:
        raise OSError(f"{description} could not be read at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return value


def _weight_names_from_index(index_path: Path) -> set[str]:
    index = _read_json_object(index_path, "model weight index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not all(
        (isinstance(name, str) for name in weight_map.values())
    ):
        raise ValueError(f"model weight index has invalid weight_map: {index_path}")
    return set(weight_map.values())


def _find_weight_paths(root: Path, index_weight_names: set[str]) -> list[Path]:
    if index_weight_names:
        paths = [
            _contained_index_weight_path(root, name)
            for name in sorted(index_weight_names)
        ]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "local model weight listed by index does not exist: "
                + ", ".join((str(path) for path in missing))
            )
        return paths
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix in _WEIGHT_SUFFIXES
        ),
        key=lambda path: str(path.relative_to(root)),
    )


def _contained_index_weight_path(root: Path, indexed_name: str) -> Path:
    candidate = Path(indexed_name)
    if candidate.is_absolute():
        raise ValueError(
            f"model weight index path escapes the local model root: {indexed_name!r}"
        )
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"model weight index path escapes the local model root: {indexed_name!r}"
        ) from exc
    return resolved


def _find_local_revision(config: Mapping[str, object]) -> str | None:
    for field in ("_commit_hash", "commit_hash", "revision", "model_revision"):
        revision = config.get(field)
        if isinstance(revision, str) and revision:
            return revision
    return None


def _probe_import(package_name: str) -> dict[str, object]:
    try:
        spec = importlib.util.find_spec(package_name)
    except Exception as exc:
        return {
            "available": False,
            "availability": "import_error",
            "version": None,
            "path": None,
            "error": f"find_spec failed: {type(exc).__name__}: {exc}",
        }
    if spec is None:
        return {
            "available": False,
            "availability": "unavailable",
            "version": None,
            "path": None,
            "error": None,
        }
    try:
        module = importlib.import_module(package_name)
    except Exception as exc:
        return {
            "available": False,
            "availability": "import_error",
            "version": None,
            "path": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    version = getattr(module, "__version__", None)
    module_path = getattr(module, "__file__", None)
    return {
        "available": True,
        "availability": "available",
        "version": str(version) if version is not None else None,
        "path": str(Path(module_path).resolve()) if module_path else None,
        "error": None,
    }


def _collect_cuda_report(torch_report: Mapping[str, object]) -> dict[str, object]:
    if not torch_report["available"]:
        return {
            "available": False,
            "device_count": 0,
            "bf16_supported": False,
            "available_memory_bytes": None,
            "devices": [],
            "reason": "torch is not installed",
            "probe_errors": [],
        }
    try:
        if importlib.util.find_spec("torch") is None:
            return {
                "available": False,
                "device_count": 0,
                "bf16_supported": False,
                "available_memory_bytes": None,
                "devices": [],
                "reason": "torch became unavailable before CUDA probing",
                "probe_errors": ["import: torch spec is unavailable"],
            }
        torch = importlib.import_module("torch")
    except Exception as exc:
        return {
            "available": False,
            "device_count": 0,
            "bf16_supported": False,
            "available_memory_bytes": None,
            "devices": [],
            "reason": f"torch CUDA import failed: {type(exc).__name__}: {exc}",
            "probe_errors": [f"import: {type(exc).__name__}: {exc}"],
        }
    try:
        cuda = torch.cuda
        available = bool(cuda.is_available())
    except Exception as exc:
        error = f"is_available: {type(exc).__name__}: {exc}"
        return {
            "available": False,
            "device_count": 0,
            "bf16_supported": False,
            "available_memory_bytes": None,
            "devices": [],
            "reason": f"torch CUDA probe failed: {error}",
            "probe_errors": [error],
        }
    if not available:
        return {
            "available": False,
            "device_count": 0,
            "bf16_supported": False,
            "available_memory_bytes": None,
            "devices": [],
            "reason": "CUDA is not available",
            "probe_errors": [],
        }
    probe_errors: list[str] = []
    device_count: int | None
    try:
        device_count = int(cuda.device_count())
    except Exception as exc:
        device_count = None
        probe_errors.append(f"device_count: {type(exc).__name__}: {exc}")
    devices: list[dict[str, object]] = []
    available_memory: int | None = None
    for device_index in range(device_count or 0):
        try:
            properties = cuda.get_device_properties(device_index)
            devices.append(
                {
                    "index": device_index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
        except Exception as exc:
            probe_errors.append(f"device[{device_index}]: {type(exc).__name__}: {exc}")
    try:
        available_memory = int(cuda.mem_get_info()[0])
    except Exception as exc:
        available_memory = None
        probe_errors.append(f"mem_get_info: {type(exc).__name__}: {exc}")
    try:
        bf16_supported = bool(cuda.is_bf16_supported())
    except Exception as exc:
        bf16_supported = False
        probe_errors.append(f"is_bf16_supported: {type(exc).__name__}: {exc}")
    return {
        "available": True,
        "device_count": device_count,
        "bf16_supported": bf16_supported,
        "available_memory_bytes": available_memory,
        "devices": devices,
        "reason": (
            None
            if not probe_errors
            else "torch CUDA probe errors: " + "; ".join(probe_errors)
        ),
        "probe_errors": probe_errors,
    }


def _resolve_project_paths(
    project_paths: Mapping[str, str | Path] | None,
) -> dict[str, str]:
    if project_paths is None:
        experiment_root = Path(__file__).resolve().parents[1]
        project_paths = {
            "experiment": experiment_root,
            "oal_attention": experiment_root,
        }
    return {
        name: str(Path(path).resolve())
        for (name, path) in sorted(project_paths.items())
    }
