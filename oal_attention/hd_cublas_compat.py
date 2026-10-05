"""Backend identity and process-local state for private HD contractions.

This module deliberately performs no compilation or implicit library loading.
Evidence is installed explicitly before plan construction; hot execution later
receives both the stable plan identity and a process-local loaded token.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
from threading import RLock
from typing import Any

import torch

from .hd_block_gemm_profiling import _record_hd_stage

_IDENTITY_SCHEMA = "hd_contraction_backend_identity_v1"
_NATIVE_PROBE_SCHEMA = "hd_bf16_bmm_probe_v1"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_NATIVE_PROBE_SOURCE = (
    Path(__file__).resolve().parent / "build" / "probe_hd_bf16_bmm.py"
)
_BACKEND_KINDS = frozenset(
    ("fp32_ieee", "torch_bmm_out_dtype", "cublas_strided_batched_ex")
)
_HEX_DIGITS = frozenset("0123456789abcdef")
_RUNTIME_SIGNATURE_SCHEMA = "hd_cublas_contraction_signature_v1"
_LAYOUT_CODES = {"R": 0, "T": 1}
RuntimeContractionSignature = tuple[int, ...]
DispatchContractionSignature = tuple[int, ...]


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True)
class HdContractionBackendIdentity:
    """Serializable backend identity stored inside an immutable HD plan."""

    backend_kind: str
    artifact_identity_sha256: str
    runtime_probe_identity_sha256: str
    capability_table_identity_sha256: str
    device_capability_class: str
    identity_schema_version: str = _IDENTITY_SCHEMA
    identity_sha256: str = ""
    signature_backend_identity_sha256: str = ""

    def __post_init__(self) -> None:
        if self.backend_kind not in _BACKEND_KINDS:
            raise ValueError("unsupported HD contraction backend kind")
        if self.identity_schema_version != _IDENTITY_SCHEMA:
            raise ValueError("unsupported HD contraction identity schema")
        if self.backend_kind == "cublas_strided_batched_ex":
            for name in (
                "artifact_identity_sha256",
                "runtime_probe_identity_sha256",
                "capability_table_identity_sha256",
                "device_capability_class",
            ):
                _require_sha256(getattr(self, name), name=name)
        expected = _canonical_sha256(self.payload())
        if self.identity_sha256 and self.identity_sha256 != expected:
            raise ValueError("HD contraction backend identity hash is invalid")
        object.__setattr__(self, "identity_sha256", expected)
        signature_backend_identity = _canonical_sha256(
            {
                "backend_kind": self.backend_kind,
                "artifact_identity_sha256": self.artifact_identity_sha256,
                "device_capability_class": self.device_capability_class,
                "identity_schema_version": self.identity_schema_version,
            }
        )
        if (
            self.signature_backend_identity_sha256
            and self.signature_backend_identity_sha256 != signature_backend_identity
        ):
            raise ValueError(
                "HD contraction signature backend identity hash is invalid"
            )
        object.__setattr__(
            self,
            "signature_backend_identity_sha256",
            signature_backend_identity,
        )

    @classmethod
    def fp32_ieee(cls) -> HdContractionBackendIdentity:
        return cls(
            backend_kind="fp32_ieee",
            artifact_identity_sha256="builtin:torch_bmm_fp32",
            runtime_probe_identity_sha256="builtin:not_required",
            capability_table_identity_sha256="builtin:dense_bmm",
            device_capability_class="device_agnostic",
        )

    @classmethod
    def verified_native(
        cls,
        *,
        runtime_probe_identity_sha256: str,
        capability_table_identity_sha256: str,
        device_capability_class: str,
    ) -> HdContractionBackendIdentity:
        return cls(
            backend_kind="torch_bmm_out_dtype",
            artifact_identity_sha256="builtin:torch_bmm_out_dtype",
            runtime_probe_identity_sha256=_require_sha256(
                runtime_probe_identity_sha256,
                name="runtime_probe_identity_sha256",
            ),
            capability_table_identity_sha256=_require_sha256(
                capability_table_identity_sha256,
                name="capability_table_identity_sha256",
            ),
            device_capability_class=_require_sha256(
                device_capability_class,
                name="device_capability_class",
            ),
        )

    @classmethod
    def cublas_compat(
        cls,
        *,
        artifact_identity_sha256: str,
        runtime_probe_identity_sha256: str,
        capability_table_identity_sha256: str,
        device_capability_class: str,
    ) -> HdContractionBackendIdentity:
        return cls(
            backend_kind="cublas_strided_batched_ex",
            artifact_identity_sha256=artifact_identity_sha256,
            runtime_probe_identity_sha256=runtime_probe_identity_sha256,
            capability_table_identity_sha256=capability_table_identity_sha256,
            device_capability_class=device_capability_class,
        )

    def payload(self) -> dict[str, object]:
        return {
            "backend_kind": self.backend_kind,
            "artifact_identity_sha256": self.artifact_identity_sha256,
            "runtime_probe_identity_sha256": self.runtime_probe_identity_sha256,
            "capability_table_identity_sha256": (self.capability_table_identity_sha256),
            "device_capability_class": self.device_capability_class,
            "identity_schema_version": self.identity_schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.payload(), "identity_sha256": self.identity_sha256}


@dataclass(eq=False)
class LoadedHdContractionBackendToken:
    """Process-local callable and execution state; never serialize this value."""

    identity: HdContractionBackendIdentity
    operator: Callable[..., object]
    load_generation: int
    device_index: int | None
    admitted_signature_keys: frozenset[str]
    admitted_runtime_signature_keys: frozenset[RuntimeContractionSignature] = (
        frozenset()
    )
    admitted_dispatch_signature_keys: frozenset[DispatchContractionSignature] = (
        frozenset()
    )
    execution_mode_identity: str = "builtin"
    expected_stream: int | None = None
    warmed: bool = False
    formal_verified: bool = True


@dataclass(frozen=True)
class ResolvedHdContractionBackend:
    requested_precision: str
    effective_precision: str
    identity: HdContractionBackendIdentity
    fallback_reason: str | None


_REGISTRY_LOCK = RLock()
_NATIVE_TOKENS: dict[str, LoadedHdContractionBackendToken] = {}
_COMPAT_TOKENS: dict[str, LoadedHdContractionBackendToken] = {}
_FP32_TOKENS: dict[str, LoadedHdContractionBackendToken] = {}
_LOAD_GENERATION = 0
_LOADED_ARTIFACT_IDENTITY: str | None = None
_LOADED_OPERATOR: Callable[..., object] | None = None
_LOADED_METADATA: dict[str, object] | None = None
_FP32_IDENTITY = HdContractionBackendIdentity.fp32_ieee()


def _next_load_generation() -> int:
    global _LOAD_GENERATION
    with _REGISTRY_LOCK:
        _LOAD_GENERATION += 1
        return _LOAD_GENERATION


def _fp32_operator(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    return torch.bmm(left, right, out=out)


def _native_operator(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    return torch.bmm(left, right, out_dtype=torch.float32, out=out)


def _make_builtin_token(
    identity: HdContractionBackendIdentity,
    *,
    device_index: int | None = None,
) -> LoadedHdContractionBackendToken:
    if identity.backend_kind == "fp32_ieee":
        operator = _fp32_operator
    elif identity.backend_kind == "torch_bmm_out_dtype":
        operator = _native_operator
    else:
        raise ValueError("compat identities require an explicitly loaded operator")
    return LoadedHdContractionBackendToken(
        identity=identity,
        operator=operator,
        load_generation=_next_load_generation(),
        device_index=device_index,
        admitted_signature_keys=frozenset(),
    )


def _canonical_device(device: torch.device | str) -> torch.device:
    try:
        canonical = torch.device(device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("backend device is invalid") from error
    if canonical.type != "cuda":
        return canonical
    if canonical.index is None:
        canonical = torch.device("cuda", torch.cuda.current_device())
    return canonical


def _current_device_capability(device: torch.device) -> tuple[int, int]:
    return tuple(torch.cuda.get_device_capability(device))


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read backend evidence: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("backend evidence must contain a JSON object")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"could not read backend artifact: {path}") from error
    return digest.hexdigest()


def _alignment_class(pointer: int) -> int:
    if pointer == 0:
        return 256
    return min(pointer & -pointer, 256)


def _dense_layout(
    tensor: torch.Tensor,
    *,
    rows: int,
    columns: int,
    name: str,
    allow_transpose: bool,
) -> str:
    if tensor.ndim != 3:
        raise ValueError(f"{name} must be rank 3")
    strides = tuple(int(value) for value in tensor.stride())
    row_major = strides == (rows * columns, columns, 1)
    simple_transpose = strides == (rows * columns, 1, rows)
    if row_major:
        return "R"
    if allow_transpose and simple_transpose:
        return "T"
    qualifier = (
        "dense row-major or simple-transpose" if allow_transpose else "dense row-major"
    )
    raise ValueError(f"{name} must use exact {qualifier} layout")


def _contraction_signature_payload(
    left: torch.Tensor,
    right: torch.Tensor,
    out: torch.Tensor,
    *,
    backend_identity: HdContractionBackendIdentity,
    execution_mode_identity: str,
) -> dict[str, object]:
    (
        shape,
        layouts,
        leading_dimensions,
        batch_strides,
        storage_offsets,
        pointer_mod_256,
        alignment_classes,
    ) = _contraction_signature_components(left, right, out)
    batch, rows, columns, reduction = shape
    left_layout, right_layout, out_layout = layouts
    return {
        "schema": _RUNTIME_SIGNATURE_SCHEMA,
        "shape": [batch, rows, columns, reduction],
        "layouts": [left_layout, right_layout, out_layout],
        "data_types": ["bfloat16", "bfloat16", "float32"],
        "scalar_type": "float32",
        "compute_type": "CUBLAS_COMPUTE_32F",
        "ops": [
            "N" if right_layout == "R" else "T",
            "N" if left_layout == "R" else "T",
        ],
        "algorithm": "CUBLAS_GEMM_DEFAULT",
        "leading_dimensions": list(leading_dimensions),
        "batch_strides": list(batch_strides),
        "storage_offsets": list(storage_offsets),
        "pointer_mod_256": list(pointer_mod_256),
        "alignment_classes": list(alignment_classes),
        "execution_mode_identity": execution_mode_identity,
        "provider_backend_identity": backend_identity.signature_backend_identity_sha256,
        "device_capability_class": backend_identity.device_capability_class,
    }


def _contraction_signature_components(
    left: torch.Tensor,
    right: torch.Tensor,
    out: torch.Tensor,
) -> tuple[
    tuple[int, int, int, int],
    tuple[str, str, str],
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
]:
    """Validate one BMM and return only its runtime-varying signature fields."""
    if not all(isinstance(tensor, torch.Tensor) for tensor in (left, right, out)):
        raise TypeError("contraction operands and output must be tensors")
    if left.ndim != 3 or right.ndim != 3 or out.ndim != 3:
        raise ValueError("contraction tensors must be rank 3")
    batch, rows, reduction = (int(value) for value in left.shape)
    right_batch, right_reduction, columns = (int(value) for value in right.shape)
    if min(batch, rows, reduction, columns) <= 0:
        raise ValueError("zero-size contractions are unsupported")
    if (right_batch, right_reduction) != (batch, reduction):
        raise ValueError("right shape does not match left")
    if tuple(out.shape) != (batch, rows, columns):
        raise ValueError("out shape does not match the contraction")
    if left.device != right.device or left.device != out.device:
        raise ValueError("contraction tensors must use the same device")
    if left.dtype != torch.bfloat16 or right.dtype != torch.bfloat16:
        raise TypeError("bf16_tensorcore operands must be bfloat16")
    if out.dtype != torch.float32:
        raise TypeError("bf16_tensorcore output must be float32")
    left_layout = _dense_layout(
        left,
        rows=rows,
        columns=reduction,
        name="left",
        allow_transpose=True,
    )
    right_layout = _dense_layout(
        right,
        rows=reduction,
        columns=columns,
        name="right",
        allow_transpose=True,
    )
    out_layout = _dense_layout(
        out,
        rows=rows,
        columns=columns,
        name="out",
        allow_transpose=False,
    )
    pointers = tuple(int(tensor.data_ptr()) for tensor in (right, left, out))
    return (
        (batch, rows, columns, reduction),
        (left_layout, right_layout, out_layout),
        (
            columns if right_layout == "R" else reduction,
            reduction if left_layout == "R" else rows,
            columns,
        ),
        (
            int(right.stride(0)),
            int(left.stride(0)),
            int(out.stride(0)),
        ),
        (
            int(right.storage_offset()),
            int(left.storage_offset()),
            int(out.storage_offset()),
        ),
        tuple(pointer % 256 for pointer in pointers),
        tuple(_alignment_class(pointer) for pointer in pointers),
    )


def _contraction_runtime_signature_key(
    left: torch.Tensor,
    right: torch.Tensor,
    out: torch.Tensor,
) -> RuntimeContractionSignature:
    """Build the compact numeric, hash-free live key used by the facade."""
    (
        shape,
        layouts,
        leading_dimensions,
        batch_strides,
        storage_offsets,
        pointer_mod_256,
        alignment_classes,
    ) = _contraction_signature_components(left, right, out)
    return (
        *shape,
        *(_LAYOUT_CODES[layout] for layout in layouts),
        *leading_dimensions,
        *batch_strides,
        *storage_offsets,
        *pointer_mod_256,
        *alignment_classes,
    )


def _contraction_signature_key(
    left: torch.Tensor,
    right: torch.Tensor,
    out: torch.Tensor,
    *,
    backend_identity: HdContractionBackendIdentity,
    execution_mode_identity: str,
) -> str:
    return _canonical_sha256(
        _contraction_signature_payload(
            left,
            right,
            out,
            backend_identity=backend_identity,
            execution_mode_identity=execution_mode_identity,
        )
    )


def _validate_loaded_token_for_contraction(
    backend_identity: HdContractionBackendIdentity,
    loaded_backend_token: LoadedHdContractionBackendToken,
    left: torch.Tensor,
    right: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Perform the small fail-closed checks that remain on the live facade."""
    if loaded_backend_token.identity != backend_identity:
        raise RuntimeError("loaded contraction backend identity does not match plan")
    if loaded_backend_token.device_index != left.device.index:
        raise RuntimeError(
            "loaded contraction backend token does not match tensor device"
        )
    if backend_identity.backend_kind == "cublas_strided_batched_ex":
        # Validate the actual BMM contract, not a benchmark's shape allowlist.
        _contraction_signature_components(left, right, out)
    elif backend_identity.backend_kind != "torch_bmm_out_dtype":
        raise RuntimeError("BF16 plan selected an invalid contraction backend")


def _current_provider_identity(device: torch.device) -> dict[str, object]:
    torch.cuda.set_device(device)
    left = torch.ones((1, 2, 2), dtype=torch.float32, device=device)
    right = torch.ones((1, 2, 2), dtype=torch.float32, device=device)
    torch.bmm(left, right)
    torch.cuda.synchronize(device)
    del left, right
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        raise RuntimeError("cuBLAS provider discovery requires Linux /proc/self/maps")
    providers: set[Path] = set()
    cuda_runtimes: set[Path] = set()
    for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
        candidate = line.rsplit(maxsplit=1)[-1].removesuffix(" (deleted)")
        path = Path(candidate)
        if path.is_file():
            if "libcublas.so" in candidate and "libcublasLt" not in candidate:
                providers.add(path.resolve())
            elif "libcudart.so" in candidate:
                cuda_runtimes.add(path.resolve())
    if len(providers) != 1:
        raise RuntimeError(
            f"expected one loaded Torch cuBLAS provider, found {tuple(providers)}"
        )
    provider = next(iter(providers))
    if len(cuda_runtimes) != 1:
        raise RuntimeError(
            "expected one loaded Torch CUDA runtime provider, "
            f"found {tuple(cuda_runtimes)}"
        )
    cuda_runtime = next(iter(cuda_runtimes))
    match = re.match(r"(libcublas\.so\.\d+)", provider.name)
    return {
        "provider_realpath": str(provider),
        "provider_sha256": _file_sha256(provider),
        "provider_soname": match.group(1) if match is not None else provider.name,
        "cuda_runtime_realpath": str(cuda_runtime),
        "cuda_runtime_sha256": _file_sha256(cuda_runtime),
    }


def _load_operator_library(
    library: Path,
) -> tuple[Callable[..., object], dict[str, object]]:
    torch.ops.load_library(str(library.resolve()))
    operator = torch.ops.hd_cublas_compat.bmm_fp32
    metadata = json.loads(torch.ops.hd_cublas_compat.runtime_metadata())
    if not isinstance(metadata, dict):
        raise RuntimeError("HD cuBLAS runtime metadata is invalid")
    return operator, metadata


def _validate_hashed_payload(payload: Mapping[str, object], *, field: str) -> str:
    identity = payload.get(field)
    unsigned = {key: value for key, value in payload.items() if key != field}
    if not isinstance(identity, str) or identity != _canonical_sha256(unsigned):
        raise ValueError(f"{field} is invalid")
    return identity


def _runtime_probe_identity(
    payload: Mapping[str, object],
    *,
    identity_policy: str,
) -> str:
    """Resolve a probe locator while retaining strict manifest/ABI validation."""
    if identity_policy not in ("semantic_compat", "legacy_strict"):
        raise ValueError("unsupported identity policy")
    if identity_policy == "legacy_strict":
        return _validate_hashed_payload(payload, field="identity_sha256")
    unsigned = {
        key: value for key, value in payload.items() if key != "identity_sha256"
    }
    return _canonical_sha256(unsigned)


def _validate_artifact_manifest(
    manifest: Mapping[str, object],
    *,
    provider_identity: Mapping[str, object],
) -> tuple[Path, str, Mapping[str, object]]:
    if manifest.get("schema_version") != "hd_cublas_compat_artifact_v1":
        raise ValueError("HD cuBLAS artifact schema is unsupported")
    compatibility = manifest.get("load_compatibility")
    if not isinstance(compatibility, Mapping):
        raise ValueError("HD cuBLAS artifact compatibility metadata is incomplete")
    # Build/source hashes are provenance, not execution licenses. Binary integrity
    # and the live Torch/CUDA ABI below are the load boundary.
    expected_runtime = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "machine": platform.machine(),
    }
    if not isinstance(compatibility.get("platform"), str) or not compatibility.get(
        "platform"
    ):
        raise ValueError("HD cuBLAS artifact platform metadata is incomplete")
    if not isinstance(
        compatibility.get("cuda_runtime_soname"), str
    ) or not compatibility.get("cuda_runtime_soname"):
        raise ValueError("HD cuBLAS artifact CUDA runtime metadata is incomplete")
    for name, expected in expected_runtime.items():
        if compatibility.get(name) != expected:
            raise ValueError(f"HD cuBLAS artifact {name} does not match")
    for name in ("provider_soname",):
        if compatibility.get(name) != provider_identity.get(name):
            raise ValueError(f"HD cuBLAS artifact {name} does not match")
    library_value = manifest.get("library_path")
    if not isinstance(library_value, str):
        raise ValueError("HD cuBLAS artifact library path is missing")
    library = Path(library_value).resolve()
    if manifest.get("shared_object_sha256") != _file_sha256(library):
        raise ValueError("HD cuBLAS shared-object hash is stale")
    artifact_identity = _require_sha256(
        manifest.get("artifact_identity_sha256"),
        name="artifact_identity_sha256",
    )
    return library, artifact_identity, compatibility


def _validate_compat_probe(
    probe: Mapping[str, object],
    *,
    artifact_identity: str,
    device: torch.device,
    identity_policy: str,
) -> tuple[str, str, str, frozenset[str]]:
    if probe.get("schema_version") != _NATIVE_PROBE_SCHEMA:
        raise ValueError("HD cuBLAS runtime probe schema is unsupported")
    if probe.get("status") != "verified_cublas_compat":
        raise ValueError("HD cuBLAS runtime probe is not verified")
    if probe.get("artifact_identity_sha256") != artifact_identity:
        raise ValueError("HD cuBLAS runtime probe artifact does not match")
    runtime_probe_identity = _runtime_probe_identity(
        probe,
        identity_policy=identity_policy,
    )
    environment = probe.get("environment")
    device_payload = probe.get("device")
    if not isinstance(environment, Mapping) or not isinstance(device_payload, Mapping):
        raise ValueError("HD cuBLAS runtime probe environment is incomplete")
    if environment.get("torch_version") != torch.__version__:
        raise ValueError("HD cuBLAS runtime probe Torch version does not match")
    if environment.get("cuda_version") != torch.version.cuda:
        raise ValueError("HD cuBLAS runtime probe CUDA version does not match")
    if tuple(device_payload.get("capability", ())) != _current_device_capability(
        device
    ):
        raise ValueError("HD cuBLAS runtime probe capability does not match")
    capability_table = _require_sha256(
        probe.get("capability_table_identity_sha256"),
        name="capability_table_identity_sha256",
    )
    capability_class = _require_sha256(
        probe.get("device_capability_class"),
        name="device_capability_class",
    )
    admitted = probe.get("admitted_signature_keys")
    if not isinstance(admitted, list) or any(
        not isinstance(value, str) for value in admitted
    ):
        raise ValueError("HD cuBLAS admitted signature table is invalid")
    return (
        runtime_probe_identity,
        capability_table,
        capability_class,
        frozenset(admitted),
    )


def load_hd_cublas_compat(
    manifest_path: str | Path,
    runtime_probe_path: str | Path | None,
    device: torch.device | str,
    *,
    identity_policy: str = "semantic_compat",
) -> LoadedHdContractionBackendToken:
    """Load an ABI-compatible artifact; an optional probe supplies diagnostics.

    Exact benchmark signatures and formal witness bundles are not runtime
    permissions. The native extension validates every live contraction.
    """
    global _LOADED_ARTIFACT_IDENTITY, _LOADED_METADATA, _LOADED_OPERATOR
    canonical_device = _canonical_device(device)
    if canonical_device.type != "cuda":
        raise ValueError("HD cuBLAS compatibility backend requires CUDA")
    provider_identity = _current_provider_identity(canonical_device)
    manifest = _load_json_object(Path(manifest_path))
    library, artifact_identity, compatibility = _validate_artifact_manifest(
        manifest,
        provider_identity=provider_identity,
    )
    with _REGISTRY_LOCK:
        if (
            _LOADED_ARTIFACT_IDENTITY is not None
            and _LOADED_ARTIFACT_IDENTITY != artifact_identity
        ):
            raise RuntimeError(
                "a different HD cuBLAS compatibility artifact is already loaded"
            )

    probe: Mapping[str, object] | None = None
    if runtime_probe_path is None:
        runtime_probe_identity = "0" * 64
        capability_table = "0" * 64
        capability_class = _canonical_sha256(
            {
                "schema": "hd_cublas_probe_only_capability_v1",
                "capability": list(_current_device_capability(canonical_device)),
                "provider": dict(provider_identity),
            }
        )
        admitted = frozenset()
        formal_verified = True
    else:
        probe = _load_json_object(Path(runtime_probe_path))
        (
            runtime_probe_identity,
            capability_table,
            capability_class,
            admitted,
        ) = _validate_compat_probe(
            probe,
            artifact_identity=artifact_identity,
            device=canonical_device,
            identity_policy=identity_policy,
        )
        formal_verified = True

    with _REGISTRY_LOCK:
        if _LOADED_OPERATOR is None or _LOADED_METADATA is None:
            operator, metadata = _load_operator_library(library)
            observed_provider = metadata.get("provider_realpath")
            if (
                not isinstance(observed_provider, str)
                or Path(observed_provider).resolve()
                != Path(str(provider_identity["provider_realpath"])).resolve()
            ):
                raise RuntimeError("post-load cuBLAS provider does not match")
            if metadata.get("runtime_cublas_version") != compatibility.get(
                "runtime_cublas_version"
            ):
                raise RuntimeError("post-load cuBLAS runtime version does not match")
            _LOADED_OPERATOR = operator
            _LOADED_METADATA = dict(metadata)
            _LOADED_ARTIFACT_IDENTITY = artifact_identity
        operator = _LOADED_OPERATOR
        metadata = _LOADED_METADATA
    if operator is None or metadata is None:
        raise RuntimeError("HD cuBLAS compatibility operator was not loaded")

    execution_mode_identity = _canonical_sha256(
        {
            "schema": "hd_cublas_execution_mode_v1",
            "policy": "single_current_stream_nonconcurrent_v1",
            "workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "<unset>"),
            "math_mode": metadata.get("math_mode"),
            "pointer_mode": metadata.get("pointer_mode"),
        }
    )
    identity = HdContractionBackendIdentity.cublas_compat(
        artifact_identity_sha256=artifact_identity,
        runtime_probe_identity_sha256=runtime_probe_identity,
        capability_table_identity_sha256=capability_table,
        device_capability_class=capability_class,
    )
    token = LoadedHdContractionBackendToken(
        identity=identity,
        operator=operator,
        load_generation=_next_load_generation(),
        device_index=canonical_device.index,
        admitted_signature_keys=admitted,
        execution_mode_identity=execution_mode_identity,
        formal_verified=formal_verified,
    )
    key = str(canonical_device)
    with _REGISTRY_LOCK:
        existing = _COMPAT_TOKENS.get(key)
        if existing is not None and existing.identity != identity:
            raise RuntimeError("a different compat token is already loaded for device")
        _COMPAT_TOKENS[key] = token if existing is None else existing
        return _COMPAT_TOKENS[key]


def register_verified_native_probe(
    probe_path: str | Path,
    device: torch.device | str,
    *,
    identity_policy: str = "semantic_compat",
) -> LoadedHdContractionBackendToken:
    """Install source-bound native evidence before any BF16 plan is built."""
    canonical_device = _canonical_device(device)
    if canonical_device.type != "cuda":
        raise ValueError("native BF16 backend requires a CUDA device")
    payload = _load_json_object(Path(probe_path))
    if payload.get("schema_version") != _NATIVE_PROBE_SCHEMA:
        raise ValueError("native contraction probe schema is unsupported")
    if payload.get("status") != "verified_vendor":
        raise ValueError("native contraction probe is not verified_vendor")
    if not isinstance(payload.get("probe_source_sha256"), str):
        raise ValueError("native contraction probe source provenance is invalid")
    identity_sha256 = _runtime_probe_identity(
        payload,
        identity_policy=identity_policy,
    )
    environment = payload.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("native contraction probe environment is missing")
    if environment.get("torch_version") != torch.__version__:
        raise ValueError("native contraction probe Torch version does not match")
    if environment.get("cuda_version") != torch.version.cuda:
        raise ValueError("native contraction probe CUDA version does not match")
    device_payload = payload.get("device")
    if not isinstance(device_payload, Mapping):
        raise ValueError("native contraction probe device is missing")
    capability = tuple(device_payload.get("capability", ()))
    if capability != _current_device_capability(canonical_device):
        raise ValueError("native contraction probe capability does not match")
    capability_table_identity = _canonical_sha256(
        {
            "schema": "hd_native_capability_table_v1",
            "cases": payload.get("cases"),
            "kernel_names": payload.get("kernel_names"),
            "decision_thresholds": payload.get("decision_thresholds"),
        }
    )
    capability_class = _canonical_sha256(
        {
            "schema": "hd_native_device_capability_v1",
            "capability": list(capability),
            "device_name": device_payload.get("name"),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
    )
    identity = HdContractionBackendIdentity.verified_native(
        runtime_probe_identity_sha256=identity_sha256,
        capability_table_identity_sha256=capability_table_identity,
        device_capability_class=capability_class,
    )
    token = _make_builtin_token(identity, device_index=canonical_device.index)
    key = str(canonical_device)
    with _REGISTRY_LOCK:
        existing = _NATIVE_TOKENS.get(key)
        if existing is not None and existing.identity != identity:
            raise RuntimeError("a different native contraction probe is already loaded")
        _NATIVE_TOKENS[key] = token if existing is None else existing
        return _NATIVE_TOKENS[key]


def resolve_backend_for_plan(
    requested_precision: str,
    device: torch.device | str,
    *,
    strict_backend: bool,
) -> ResolvedHdContractionBackend:
    if requested_precision == "fp32_ieee":
        return ResolvedHdContractionBackend(
            requested_precision=requested_precision,
            effective_precision="fp32_ieee",
            identity=_FP32_IDENTITY,
            fallback_reason=None,
        )
    if requested_precision != "bf16_tensorcore":
        raise ValueError("unsupported HD Block-GEMM precision")
    canonical_device = _canonical_device(device)
    if canonical_device.type != "cuda":
        raise ValueError("bf16_tensorcore requires CUDA")
    key = str(canonical_device)
    with _REGISTRY_LOCK:
        token = _COMPAT_TOKENS.get(key) or _NATIVE_TOKENS.get(key)
    if token is not None and token.formal_verified:
        return ResolvedHdContractionBackend(
            requested_precision=requested_precision,
            effective_precision="bf16_tensorcore",
            identity=token.identity,
            fallback_reason=None,
        )
    if strict_backend:
        raise RuntimeError(
            "bf16_tensorcore requires an explicitly registered verified backend"
        )
    return ResolvedHdContractionBackend(
        requested_precision=requested_precision,
        effective_precision="fp32_ieee",
        identity=_FP32_IDENTITY,
        fallback_reason="no_verified_bf16_contraction_backend",
    )


def prepare_backend_token(
    identity: HdContractionBackendIdentity,
    device: torch.device | str,
) -> LoadedHdContractionBackendToken:
    """Bind one immutable plan identity to its process-local callable.

    This lookup belongs at prepared-context construction, never inside an
    individual contraction.  The returned object is therefore shared by all
    forward and backward contractions for the invocation.
    """
    if not isinstance(identity, HdContractionBackendIdentity):
        raise TypeError("backend identity must be an HD contraction identity")
    canonical_device = _canonical_device(device)
    key = str(canonical_device)
    with _REGISTRY_LOCK:
        if identity.backend_kind == "fp32_ieee":
            token = _FP32_TOKENS.get(key)
            if token is None:
                token = _make_builtin_token(
                    identity,
                    device_index=canonical_device.index,
                )
                _FP32_TOKENS[key] = token
        elif identity.backend_kind == "torch_bmm_out_dtype":
            token = _NATIVE_TOKENS.get(key)
        elif identity.backend_kind == "cublas_strided_batched_ex":
            token = _COMPAT_TOKENS.get(key)
        else:  # pragma: no cover - dataclass validation owns this invariant.
            token = None
        if token is None:
            raise RuntimeError(
                "plan contraction backend is not loaded for the target device"
            )
        if token.identity != identity:
            raise RuntimeError(
                "loaded contraction backend identity does not match the plan"
            )
        if not token.formal_verified:
            raise RuntimeError("plan contraction backend is not formally verified")
        return token


def _reset_backend_registry_for_tests() -> None:
    global _LOAD_GENERATION, _LOADED_ARTIFACT_IDENTITY, _LOADED_METADATA
    global _LOADED_OPERATOR
    with _REGISTRY_LOCK:
        _NATIVE_TOKENS.clear()
        _COMPAT_TOKENS.clear()
        _FP32_TOKENS.clear()
        _LOAD_GENERATION = 0
        _LOADED_ARTIFACT_IDENTITY = None
        _LOADED_OPERATOR = None
        _LOADED_METADATA = None


__all__: tuple[str, ...] = ()
