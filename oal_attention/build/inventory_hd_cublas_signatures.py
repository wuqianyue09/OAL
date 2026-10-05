#!/usr/bin/env python3
"""Record the real HD forward/VJP contraction surface for cuBLAS probing.

This is an inventory generator, not a correctness or performance admission
tool.  It deliberately uses an FP32 diagnostic contraction facade so a Torch
build lacking ``bmm(..., out_dtype=...)`` can still execute the actual HD
forward and analytic-backward call graph.  The emitted signatures retain the
BF16/FP32 operand contract and every physical layout property required by the
private cuBLAS leaf.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Iterator, Mapping, Sequence

import torch


_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from oal_attention import hd_block_gemm as _hd_plan
from oal_attention import hd_block_gemm_backward as _hd_backward
from oal_attention import hd_block_gemm_feature_context as _hd_context
from oal_attention import hd_block_gemm_runtime as _hd_runtime
from oal_attention.hd_block_gemm_adapters import _grouped_hd_block_gemm_forward
from oal_attention.hd_block_gemm_contracts import (
    canonicalize_hd_operator_config,
)
from oal_attention.hd_cublas_compat import (
    HdContractionBackendIdentity,
    LoadedHdContractionBackendToken,
    ResolvedHdContractionBackend,
    _canonical_device,
    _contraction_signature_payload,
)


_SCHEMA = "hd_cublas_signature_inventory_v4"
_SOURCE_PATHS = {
    # The inventory calls this adapter directly.  Its validation and plan
    # selection determine which contractions enter the recording facade.
    "adapters": _ROOT / "oal_attention" / "hd_block_gemm_adapters.py",
    # The adapter dispatches autograd, whose forward/backward route chooses
    # the runtime contractions and saved views below.
    "autograd": _ROOT / "oal_attention" / "hd_block_gemm_autograd.py",
    # Planned storage/layout sources can alter strides, offsets, and pointer
    # alignment in the recorded signature surface.
    "buffers": _ROOT / "oal_attention" / "hd_block_gemm_buffers.py",
    "cache": _ROOT / "oal_attention" / "hd_block_gemm_cache.py",
    "contracts": _ROOT / "oal_attention" / "hd_block_gemm_contracts.py",
    "plan_value": _ROOT / "oal_attention" / "hd_block_gemm_plan.py",
    "facade": _ROOT / "oal_attention" / "hd_block_gemm_contractions.py",
    "feature_backward": _ROOT
    / "oal_attention"
    / "hd_block_gemm_feature_backward.py",
    "feature_context": _ROOT
    / "oal_attention"
    / "hd_block_gemm_feature_context.py",
    "feature_forward": _ROOT
    / "oal_attention"
    / "hd_block_gemm_feature_forward.py",
    "feature_ops": _ROOT / "oal_attention" / "hd_block_gemm_feature_ops.py",
    "feature_kernels": _ROOT
    / "oal_attention"
    / "triton"
    / "hd_block_gemm_feature_kernels.py",
    "key_ops": _ROOT / "oal_attention" / "hd_block_gemm_key_ops.py",
    "key_kernels": _ROOT
    / "oal_attention"
    / "triton"
    / "hd_block_gemm_key_kernels.py",
    "normalization_ops": _ROOT
    / "oal_attention"
    / "hd_block_gemm_normalization_ops.py",
    "normalization_kernels": _ROOT
    / "oal_attention"
    / "triton"
    / "hd_block_gemm_normalization_kernels.py",
    "grouped_coefficients": _ROOT / "oal_attention" / "grouped_quadratic.py",
    "inventory": Path(__file__),
    "runtime": _ROOT / "oal_attention" / "hd_block_gemm_runtime.py",
    "backward": _ROOT / "oal_attention" / "hd_block_gemm_backward.py",
    "backward_wave": _ROOT
    / "oal_attention"
    / "hd_block_gemm_backward_wave.py",
    "plan": _ROOT / "oal_attention" / "hd_block_gemm.py",
    # This owns the canonical contraction-signature schema being recorded.
    "signature_contract": _ROOT / "oal_attention" / "hd_cublas_compat.py",
}
_DEFAULT_SEQUENCE_LENGTHS = (4096, 8192, 16384, 32768)
_INVENTORY_EXECUTION_MODE = "inventory_fp32_diagnostic_single_current_stream_v1"
_CANDIDATE_SCHEDULE_SCHEMA = "hd_cublas_candidate_schedule_v1"
_DEFAULT_TOKEN_BLOCKS = (64, 128)
_DEFAULT_WAVE_OPTIONS: tuple[int | str, ...] = (4, 8, 16, "number_blocks")
_GENERIC_OPERATOR_CONFIG = {
    "schema_version": "hd_query_operator_config_v4",
    "query_feature_impl": "generic_materialized",
    "query_fold_impl": "generic_materialized",
    "query_gradient_flow": "materialized",
    "query_producer_fold_strategy": "none",
    "query_consumer_stages": [],
}


def canonicalize_query_operator_config(config: object) -> dict[str, object]:
    """Canonicalize query selectors plus independent key/physical selectors."""
    return canonicalize_hd_operator_config(config)


def _stable_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _parse_operator_config(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return canonicalize_query_operator_config(payload)


def _payload_sha256(payload: Mapping[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return _sha256_bytes(_stable_json(unsigned).encode("utf-8"))


def _parse_sequence_lengths(value: str) -> tuple[int, ...]:
    try:
        lengths = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("sequence lengths must be positive")
    if len(set(lengths)) != len(lengths):
        raise argparse.ArgumentTypeError("sequence lengths must be unique")
    return tuple(sorted(lengths))


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(item <= 0 for item in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must be unique positive integers")
    return tuple(sorted(values))


def _parse_wave_options(value: str) -> tuple[int | str, ...]:
    options: list[int | str] = []
    for item in value.split(","):
        normalized = item.strip().lower()
        if normalized == "nb":
            normalized = "number_blocks"
        if normalized.startswith("runtime:"):
            try:
                cap = int(normalized.split(":", 1)[1])
            except ValueError as error:
                raise argparse.ArgumentTypeError("runtime wave cap must be a positive integer") from error
            if cap <= 0:
                raise argparse.ArgumentTypeError("runtime wave cap must be a positive integer")
            option: int | str = f"runtime:{cap}"
        elif normalized == "number_blocks":
            option = normalized
        else:
            try:
                option = int(normalized)
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    "wave options must be positive integers or nb"
                ) from error
            if option <= 0:
                raise argparse.ArgumentTypeError(
                    "wave options must be positive integers or nb"
                )
        if option in options:
            raise argparse.ArgumentTypeError("wave options must be unique")
        options.append(option)
    if not options:
        raise argparse.ArgumentTypeError("wave options must not be empty")
    return tuple(options)


def _validate_head_geometry(query_heads: int, key_value_heads: int) -> None:
    if any(
        not isinstance(heads, int) or isinstance(heads, bool) or heads <= 0
        for heads in (query_heads, key_value_heads)
    ):
        raise ValueError("query heads and key/value heads must be positive integers")
    if query_heads % key_value_heads:
        raise ValueError("query heads must be divisible by key/value heads")


def _fixture_payload(
    *, query_heads: int = 14, key_value_heads: int = 2,
) -> dict[str, object]:
    _validate_head_geometry(query_heads, key_value_heads)
    return {
        "batch_size": 1,
        "query_heads": query_heads,
        "key_value_heads": key_value_heads,
        "head_dimension": 64,
        "value_dimension": 64,
        "dtype": "bfloat16",
        "dim_groups_rule": "d_mod_2",
        "shared_factor": [1.1, 0.5, 0.9, 0.4, 0.8, 0.7],
        "factor_dtype": "float32",
        "kernel_epsilon": 1e-6,
        "scale": 0.125,
        "gradient_targets": ["q", "k", "v", "factor"],
    }


def _source_hashes() -> dict[str, str]:
    return {
        name: _sha256_bytes(path.read_bytes()) for name, path in _SOURCE_PATHS.items()
    }


def _role_counts(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("inventory signature role counts are missing")
    counts: dict[str, int] = {}
    for role, count in value.items():
        if not isinstance(role, str) or not role:
            raise ValueError("inventory signature role is invalid")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("inventory signature role count is invalid")
        counts[role] = count
    return {role: counts[role] for role in sorted(counts)}


def _canonical_signatures(
    signatures: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    canonical: dict[str, dict[str, object]] = {}
    for raw_record in signatures:
        record = dict(raw_record)
        identity = record.get("signature_identity_sha256")
        signature = record.get("signature")
        if not isinstance(identity, str) or not isinstance(signature, Mapping):
            raise ValueError("inventory signature record is incomplete")
        if identity != _sha256_bytes(_stable_json(dict(signature)).encode("utf-8")):
            raise ValueError("inventory signature identity is invalid")
        role_counts = _role_counts(record.get("role_counts"))
        existing = canonical.get(identity)
        if existing is None:
            canonical[identity] = {
                "signature_identity_sha256": identity,
                "signature": dict(signature),
                "role_counts": dict(role_counts),
            }
            continue
        if existing["signature"] != dict(signature):
            raise ValueError("inventory signature identity is not canonical")
        merged = dict(existing["role_counts"])
        for role, count in role_counts.items():
            merged[role] = int(merged.get(role, 0)) + count
        existing["role_counts"] = {role: merged[role] for role in sorted(merged)}
    return [canonical[identity] for identity in sorted(canonical)]


def _candidate_identity(
    *,
    sequence_length: int,
    token_block: int,
    feature_wave_blocks: int,
    operator_config: Mapping[str, object],
) -> str:
    return _sha256_bytes(
        _stable_json(
            {
                "schema": _CANDIDATE_SCHEDULE_SCHEMA,
                "sequence_length": sequence_length,
                "token_block": token_block,
                "feature_wave_blocks": feature_wave_blocks,
                "precision": "bf16_tensorcore",
                "result_contract": "full_aux",
                "local_mode": "generic_packed",
                "operator_config": dict(operator_config),
            }
        ).encode("utf-8")
    )


def _candidate_key(
    *,
    sequence_length: int,
    token_block: int,
    feature_wave_blocks: int,
    operator_config: Mapping[str, object],
) -> dict[str, object]:
    """Return the readable structural key emitted by new inventories."""
    return {
        "sequence_length": sequence_length,
        "token_block": token_block,
        "feature_wave_blocks": feature_wave_blocks,
        "operator_config": dict(operator_config),
    }




def _candidate_schedules(
    sequence_lengths: Sequence[int],
    *,
    token_blocks: Sequence[int],
    wave_options: Sequence[int | str],
    operator_config: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    resolved_operator_config = canonicalize_query_operator_config(
        _GENERIC_OPERATOR_CONFIG if operator_config is None else operator_config
    )
    candidates: list[dict[str, object]] = []
    for sequence_length in sequence_lengths:
        for token_block in token_blocks:
            number_blocks = (sequence_length + token_block - 1) // token_block
            for option in wave_options:
                feature_wave_blocks = (
                    number_blocks if option == "number_blocks" else option
                )
                if isinstance(option, str) and option.startswith("runtime:"):
                    feature_wave_blocks = min(number_blocks, int(option.split(":", 1)[1]))
                if not isinstance(feature_wave_blocks, int) or feature_wave_blocks <= 0:
                    raise ValueError("candidate wave option is invalid")
                if feature_wave_blocks > number_blocks:
                    continue
                candidates.append(
                    {
                        "candidate_identity_sha256": _candidate_identity(
                            sequence_length=sequence_length,
                            token_block=token_block,
                            feature_wave_blocks=feature_wave_blocks,
                            operator_config=resolved_operator_config,
                        ),
                        "sequence_length": sequence_length,
                        "token_block": token_block,
                        "feature_wave_blocks": feature_wave_blocks,
                        "candidate_key": _candidate_key(
                            sequence_length=sequence_length,
                            token_block=token_block,
                            feature_wave_blocks=feature_wave_blocks,
                            operator_config=resolved_operator_config,
                        ),
                    }
                )
    if len({candidate["candidate_identity_sha256"] for candidate in candidates}) != len(
        candidates
    ):
        raise ValueError("candidate schedule grid contains duplicates")
    return candidates


def _signature_metadata_index(operand: int) -> int:
    """Map logical BMM operands to the cuBLAS-ordered signature metadata."""
    indices = (1, 0, 2)
    try:
        return indices[operand]
    except IndexError as error:
        raise ValueError("inventory operand is invalid") from error




def _make_inventory_payload(
    *,
    device: Mapping[str, object],
    environment: Mapping[str, object],
    fixture: Mapping[str, object],
    candidate_grid: Mapping[str, object],
    candidates: Sequence[Mapping[str, object]],
    signatures: Sequence[Mapping[str, object]],
    source_hashes: Mapping[str, str],
    execution: Mapping[str, object] | None = None,
    operator_config: Mapping[str, object] | None = None,
    paired_performance_base: Mapping[str, object] | None = None,
) -> dict[str, object]:
    required_sources = set(_SOURCE_PATHS)
    if set(source_hashes) != required_sources:
        raise ValueError("inventory source hashes are incomplete")
    fixture_payload = dict(fixture)
    grid_payload = dict(candidate_grid)
    resolved_operator_config = canonicalize_query_operator_config(
        _GENERIC_OPERATOR_CONFIG if operator_config is None else operator_config
    )
    if paired_performance_base is not None:
        raise ValueError("paired_performance_base: OP-4 exploration is retired")
    canonical_signatures = _canonical_signatures(signatures)
    signature_ids = {
        str(record["signature_identity_sha256"]) for record in canonical_signatures
    }
    canonical_candidates: list[dict[str, object]] = []
    seen_candidate_ids: set[str] = set()
    for raw_candidate in candidates:
        candidate = dict(raw_candidate)
        candidate_id = candidate.get("candidate_identity_sha256")
        sequence_length = candidate.get("sequence_length")
        token_block = candidate.get("token_block")
        feature_wave_blocks = candidate.get("feature_wave_blocks")
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in (sequence_length, token_block, feature_wave_blocks)
        ) or not isinstance(candidate_id, str):
            raise ValueError("inventory candidate is incomplete")
        if candidate_id != _candidate_identity(
            sequence_length=int(sequence_length),
            token_block=int(token_block),
            feature_wave_blocks=int(feature_wave_blocks),
            operator_config=resolved_operator_config,
        ) or candidate_id in seen_candidate_ids:
            raise ValueError("inventory candidate identity is invalid")
        candidate_key = candidate.get("candidate_key")
        if candidate_key != _candidate_key(
            sequence_length=int(sequence_length),
            token_block=int(token_block),
            feature_wave_blocks=int(feature_wave_blocks),
            operator_config=resolved_operator_config,
        ):
            raise ValueError("inventory candidate key is invalid")
        execution_operator_config = candidate.get("execution_operator_config")
        if execution_operator_config != resolved_operator_config:
            raise ValueError("inventory candidate execution operator config is invalid")
        seen_candidate_ids.add(candidate_id)
        plan = candidate.get("plan")
        if (
            not isinstance(plan, Mapping)
            or plan.get("execution_operator_config") != resolved_operator_config
        ):
            raise ValueError("inventory candidate plan execution operator config is invalid")
        candidate_execution = candidate.get("execution")
        required = candidate.get("required_signature_identities")
        bmm_call_count = candidate.get("bmm_call_count")
        role_counts = _role_counts(candidate.get("bmm_role_counts"))
        if (
            not isinstance(plan.get("plan_id"), str)
            or not isinstance(candidate_execution, Mapping)
            or candidate_execution.get("forward_completed") is not True
            or candidate_execution.get("backward_completed") is not True
            or not isinstance(required, list)
            or not required
            or not all(isinstance(identity, str) for identity in required)
            or tuple(required) != tuple(sorted(required))
            or len(set(required)) != len(required)
            or not set(required) <= signature_ids
            or not isinstance(bmm_call_count, int)
            or isinstance(bmm_call_count, bool)
            or bmm_call_count <= 0
            or sum(role_counts.values()) != bmm_call_count
        ):
            raise ValueError("inventory candidate signature closure is incomplete")
        canonical_candidate = {
                "candidate_identity_sha256": candidate_id,
                "sequence_length": sequence_length,
                "token_block": token_block,
                "feature_wave_blocks": feature_wave_blocks,
                "candidate_key": dict(candidate_key),
                "execution_operator_config": dict(execution_operator_config),
                "plan": dict(plan),
                "execution": {
                    "forward_completed": True,
                    "backward_completed": True,
                },
                "required_signature_identities": list(required),
                "distinct_bmm_signature_count": len(required),
                "bmm_call_count": bmm_call_count,
                "bmm_role_counts": role_counts,
            }
        canonical_candidates.append(canonical_candidate)
    canonical_candidates.sort(
        key=lambda candidate: str(candidate["candidate_identity_sha256"])
    )
    payload: dict[str, object] = {
        "schema_version": _SCHEMA,
        "device": dict(device),
        "environment": dict(environment),
        "fixture": fixture_payload,
        "operator_config": dict(resolved_operator_config),
        "fixture_identity_sha256": _sha256_bytes(
            _stable_json(fixture_payload).encode("utf-8")
        ),
        "candidate_grid": grid_payload,
        "candidate_grid_identity_sha256": _sha256_bytes(
            _stable_json(grid_payload).encode("utf-8")
        ),
        "candidates": canonical_candidates,
        "signatures": canonical_signatures,
        "source_hashes": dict(source_hashes),
        "execution": dict(
            execution
            or {
                "forward_completed": True,
                "backward_completed": True,
                "mode": "fp32_diagnostic_emulation_inventory_only",
            }
        ),
    }
    payload["identity_sha256"] = _payload_sha256(payload)
    return payload


def _write_complete_inventory(path: Path, payload: Mapping[str, object]) -> None:
    execution = payload.get("execution")
    if not isinstance(execution, Mapping) or not (
        execution.get("forward_completed") is True
        and execution.get("backward_completed") is True
    ):
        raise RuntimeError("inventory requires a complete forward+backward execution")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("inventory requires at least one candidate closure")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _device_payload(device: torch.device) -> dict[str, object]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "type": "cuda",
        "index": index,
        "name": properties.name,
        "capability": list(torch.cuda.get_device_capability(index)),
    }


def _diagnostic_identity() -> HdContractionBackendIdentity:
    return HdContractionBackendIdentity.cublas_compat(
        artifact_identity_sha256=_sha256_bytes(b"hd_inventory_diagnostic_artifact"),
        runtime_probe_identity_sha256=_sha256_bytes(b"hd_inventory_diagnostic_probe"),
        capability_table_identity_sha256=_sha256_bytes(b"hd_inventory_diagnostic_table"),
        device_capability_class=_sha256_bytes(b"hd_inventory_diagnostic_device"),
    )


class _RecordingContractionFacade:
    """Record the private seam while using only an explicit FP32 diagnostic op."""

    def __init__(
        self,
        *,
        identity: HdContractionBackendIdentity,
        execution_mode_identity: str,
    ) -> None:
        self.identity = identity
        self.execution_mode_identity = execution_mode_identity
        self.records: list[dict[str, object]] = []

    @staticmethod
    def _caller_role() -> str:
        # These roles are generic Python BMM-facade callsites.  They make the
        # recorded call count auditable, but are not HD stage-profiler labels.
        frame = inspect.currentframe()
        # ``_caller_role`` is called from ``__call__``.  The physical HD
        # runtime/backward caller is the next frame out, not the recorder
        # itself; recording the latter would erase the useful role boundary.
        caller = frame.f_back.f_back if frame is not None and frame.f_back else None
        if caller is None:
            raise RuntimeError("inventory cannot identify contraction caller")
        module = str(caller.f_globals.get("__name__", "unknown"))
        return f"{module.rsplit('.', maxsplit=1)[-1]}.{caller.f_code.co_name}"

    def __call__(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        out: torch.Tensor,
        precision: str,
        **_: object,
    ) -> torch.Tensor:
        if precision != "bf16_tensorcore":
            raise RuntimeError("inventory must execute the BF16 HD plan")
        signature = _contraction_signature_payload(
            left,
            right,
            out,
            backend_identity=self.identity,
            execution_mode_identity=self.execution_mode_identity,
        )
        self.records.append(
            {
                "signature_identity_sha256": _sha256_bytes(
                    _stable_json(signature).encode("utf-8")
                ),
                "signature": signature,
                "role_counts": {self._caller_role(): 1},
            }
        )
        # This is intentionally not a candidate cuBLAS path: it merely keeps
        # the real call graph alive so the inventory can observe every seam.
        return torch.bmm(left.float(), right.float(), out=out)


@contextmanager
def _recording_hd_call_graph(
    recorder: _RecordingContractionFacade,
    *,
    device: torch.device,
) -> Iterator[None]:
    identity = recorder.identity
    token = LoadedHdContractionBackendToken(
        identity=identity,
        operator=lambda left, right, *, out: recorder(
            left, right, out=out, precision="bf16_tensorcore"
        ),
        load_generation=0,
        device_index=device.index,
        admitted_signature_keys=frozenset(),
        execution_mode_identity=recorder.execution_mode_identity,
        formal_verified=False,
    )
    original_resolve = _hd_plan.resolve_backend_for_plan
    original_prepare = _hd_context.prepare_backend_token
    original_runtime = _hd_runtime._bmm_fp32
    original_backward = _hd_backward._bmm_fp32

    def resolve(
        requested_precision: str,
        requested_device: torch.device | str,
        *,
        strict_backend: bool,
    ) -> ResolvedHdContractionBackend:
        del requested_device, strict_backend
        if requested_precision != "bf16_tensorcore":
            raise RuntimeError("inventory only records BF16 Tensor-Core plans")
        return ResolvedHdContractionBackend(
            requested_precision="bf16_tensorcore",
            effective_precision="bf16_tensorcore",
            identity=identity,
            fallback_reason=None,
        )

    def prepare(
        requested_identity: HdContractionBackendIdentity,
        requested_device: torch.device | str,
    ) -> LoadedHdContractionBackendToken:
        if requested_identity != identity or torch.device(requested_device) != device:
            raise RuntimeError("inventory backend token does not match the BF16 plan")
        return token

    _hd_plan.resolve_backend_for_plan = resolve
    _hd_context.prepare_backend_token = prepare
    _hd_runtime._bmm_fp32 = recorder
    _hd_backward._bmm_fp32 = recorder
    try:
        yield
    finally:
        _hd_plan.resolve_backend_for_plan = original_resolve
        _hd_context.prepare_backend_token = original_prepare
        _hd_runtime._bmm_fp32 = original_runtime
        _hd_backward._bmm_fp32 = original_backward


def _make_values(
    sequence_length: int,
    *,
    device: torch.device,
    seed: int,
    query_heads: int = 14,
    key_value_heads: int = 2,
) -> dict[str, torch.Tensor]:
    _validate_head_geometry(query_heads, key_value_heads)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn((1, query_heads, sequence_length, 64), generator=generator, dtype=torch.bfloat16)
    k = torch.randn((1, key_value_heads, sequence_length, 64), generator=generator, dtype=torch.bfloat16)
    v = torch.randn((1, key_value_heads, sequence_length, 64), generator=generator, dtype=torch.bfloat16)
    upstream = torch.randn((1, query_heads, sequence_length, 64), generator=generator, dtype=torch.bfloat16)
    return {
        "q": q.to(device).requires_grad_(),
        "k": k.to(device).requires_grad_(),
        "v": v.to(device).requires_grad_(),
        "do": upstream.to(device),
        "groups": torch.arange(64, dtype=torch.int32, device=device).remainder(2),
        "factor": torch.tensor(
            _fixture_payload()["shared_factor"], dtype=torch.float32, device=device
        ).requires_grad_(),
    }








def _run_inventory(
    *,
    device: torch.device,
    sequence_lengths: Sequence[int],
    token_blocks: Sequence[int],
    wave_options: Sequence[int | str],
    operator_config: Mapping[str, object],
    include_performance_base: bool = False,
    include_kv_cross_base: bool = False,
    include_feature_padding_base: bool = False,
    feature_padding_final_parent: bool = False,
    kv_cross_impl: str = "split",
    query_heads: int = 14,
    key_value_heads: int = 2,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if include_performance_base:
        raise ValueError("include_performance_base: OP-4 exploration is retired")
    _validate_head_geometry(query_heads, key_value_heads)
    candidates: list[dict[str, object]] = []
    signature_occurrences: list[dict[str, object]] = []
    original_build = _hd_plan.build_hd_block_gemm_plan

    captured_plans: list[dict[str, object]] = []
    def independent_execution_kwargs(
        execution_config: Mapping[str, object],
    ) -> dict[str, object]:
        kwargs = {
            "query_feature_token_tile": execution_config.get(
                "query_feature_token_tile", 1
            ),
            "query_fold_token_tile": execution_config.get(
                "query_fold_token_tile", 1
            ),
            "query_fold_input": execution_config.get(
                "query_fold_input", "staged_fp32"
            ),
            "backward_schedule": execution_config.get(
                "backward_schedule", "split"
            ),
            "gradient_staging": execution_config.get(
                "gradient_staging", "per_wave"
            ),
            "forward_normalize_impl": execution_config.get(
                "forward_normalize_impl", "torch"
            ),
            "backward_normalize_impl": execution_config.get(
                "backward_normalize_impl", "torch"
            ),
            "save_local_score": execution_config.get("save_local_score", False),
            "key_retention": execution_config.get("key_retention", "none"),
        }
        if feature_padding_final_parent:
            kwargs.update(
                {
                    "query_feature_token_tile": 4,
                    "query_fold_token_tile": 1,
                    "query_fold_input": "staged_fp32",
                    "backward_schedule": "shared_wave",
                    "gradient_staging": "per_wave",
                    "forward_normalize_impl": "triton",
                    "backward_normalize_impl": "triton",
                    "save_local_score": True,
                    "key_retention": "forward",
                }
            )
        return kwargs

    def capture_plan(*args: object, **kwargs: object) -> object:
        plan = original_build(*args, **kwargs)
        # Keep the resolved plan separate from the structural request identity.
        # execute() retains that historical identity below; it must not erase
        # evidence of selectors resolved or overridden by the actual planner.
        resolved_operator_config = {
            "schema_version": "hd_query_operator_config_v4",
            **{name: getattr(plan, name) for name in (
                "query_feature_impl", "query_fold_impl", "query_gradient_flow",
                "query_producer_fold_strategy",
            )},
            "query_consumer_stages": list(getattr(plan, "query_consumer_stages")),
            **{name: getattr(plan, name, default) for name, default in {
                "key_feature_impl": "generic_materialized",
                "key_fold_impl": "generic_materialized",
                "feature_padding": "none",
                "query_feature_token_tile": 1,
                "query_fold_token_tile": 1,
                "query_fold_input": "staged_fp32",
                "backward_schedule": "split",
                "gradient_staging": "per_wave",
                "forward_normalize_impl": "torch",
                "backward_normalize_impl": "torch",
                "kv_cross_impl": "split",
                "save_local_score": False,
                "key_retention": "none",
            }.items()},
        }
        captured_plans.append(
            {
                "resolved_plan_operator_config": resolved_operator_config,
                "sequence_length": int(getattr(plan, "sequence_length")),
                "plan_id": str(getattr(plan, "plan_id")),
                "result_contract": str(getattr(plan, "result_contract")),
                "number_blocks": int(getattr(plan, "number_blocks")),
                "kv_cross_impl": str(getattr(plan, "kv_cross_impl", "split")),
                **(
                    {"feature_padding": str(getattr(plan, "feature_padding"))}
                    if getattr(plan, "feature_padding", "none") != "none"
                    else {}
                ),
                "contraction_backend": getattr(plan, "contraction_backend_identity").to_dict(),
                "execution_operator_config": canonicalize_query_operator_config(
                    {
                        "schema_version": "hd_query_operator_config_v4",
                        "query_feature_impl": getattr(plan, "query_feature_impl"),
                        "query_fold_impl": getattr(plan, "query_fold_impl"),
                        "query_gradient_flow": getattr(plan, "query_gradient_flow"),
                        "query_producer_fold_strategy": getattr(
                            plan, "query_producer_fold_strategy"
                        ),
                        "query_consumer_stages": list(
                            getattr(plan, "query_consumer_stages")
                        ),
                        "key_feature_impl": getattr(
                            plan, "key_feature_impl", "generic_materialized"
                        ),
                        "key_fold_impl": getattr(
                            plan, "key_fold_impl", "generic_materialized"
                        ),
                        "feature_padding": getattr(plan, "feature_padding", "none"),
                    }
                ),
            }
        )
        return plan

    from oal_attention import hd_block_gemm_adapters as _adapters

    original_adapter_build = _adapters.build_hd_block_gemm_plan
    _adapters.build_hd_block_gemm_plan = capture_plan
    _hd_plan.build_hd_block_gemm_plan = capture_plan
    try:
        for schedule in _candidate_schedules(
            sequence_lengths,
            token_blocks=token_blocks,
            wave_options=wave_options,
            operator_config=operator_config,
        ):
            sequence_length = int(schedule["sequence_length"])
            def execute(
                execution_config: Mapping[str, object],
                execution_kv_cross_impl: str,
            ) -> tuple[dict[str, object], list[dict[str, object]]]:
                independent_kwargs = independent_execution_kwargs(execution_config)
                recorder = _RecordingContractionFacade(
                    identity=_diagnostic_identity(),
                    execution_mode_identity=_INVENTORY_EXECUTION_MODE,
                )
                captured_plans.clear()
                with _recording_hd_call_graph(recorder, device=device):
                    values = _make_values(
                        sequence_length,
                        device=device,
                        query_heads=query_heads,
                        key_value_heads=key_value_heads,
                        seed=20260831 + sequence_length,
                    )
                    output, _, _ = _grouped_hd_block_gemm_forward(
                        values["q"],
                        values["k"],
                        values["v"],
                        values["groups"],
                        values["factor"],
                        scale=0.125,
                        kernel_eps=1e-6,
                        token_block=int(schedule["token_block"]),
                        feature_wave_blocks=int(schedule["feature_wave_blocks"]),
                        precision="bf16_tensorcore",
                        query_feature_impl=str(execution_config["query_feature_impl"]),
                        query_fold_impl=str(execution_config["query_fold_impl"]),
                        query_gradient_flow=str(execution_config["query_gradient_flow"]),
                        query_producer_fold_strategy=str(execution_config["query_producer_fold_strategy"]),
                        query_consumer_stages=tuple(execution_config["query_consumer_stages"]),
                        key_feature_impl=str(
                            execution_config.get(
                                "key_feature_impl", "generic_materialized"
                            )
                        ),
                        key_fold_impl=str(
                            execution_config.get(
                                "key_fold_impl", "generic_materialized"
                            )
                        ),
                        kv_cross_impl=execution_kv_cross_impl,
                        feature_padding=str(
                            execution_config.get("feature_padding", "none")
                        ),
                        **independent_kwargs,
                    )
                    torch.autograd.grad(
                        output,
                        (values["q"], values["k"], values["v"], values["factor"]),
                        grad_outputs=values["do"],
                    )
                    torch.cuda.synchronize(device)
                    del values, output
                if len(captured_plans) != 1:
                    raise RuntimeError("inventory candidate did not build exactly one plan")
                candidate_plan = captured_plans[0]
                candidate_plan["execution_operator_config"] = (
                    canonicalize_query_operator_config(execution_config)
                )
                return candidate_plan, recorder.records

            candidate_plan, candidate_records = execute(
                operator_config,
                kv_cross_impl,
            )
            kv_cross_base_records: list[dict[str, object]] = []
            if include_kv_cross_base:
                _, kv_cross_base_records = execute(
                    operator_config,
                    "split",
                )
            feature_padding_base_records: list[dict[str, object]] = []
            if include_feature_padding_base:
                feature_padding_base_config = dict(operator_config)
                feature_padding_base_config.pop("feature_padding", None)
                _, feature_padding_base_records = execute(
                    feature_padding_base_config,
                    kv_cross_impl,
                )
                if include_kv_cross_base:
                    _, feature_padding_split_records = execute(
                        feature_padding_base_config,
                        "split",
                    )
                    feature_padding_base_records.extend(
                        feature_padding_split_records
                    )
            closure_records = [
                *candidate_records,
                *kv_cross_base_records,
                *feature_padding_base_records,
            ]
            required_signature_identities = sorted(
                {
                    str(record["signature_identity_sha256"])
                    for record in closure_records
                }
            )
            role_counts: dict[str, int] = {}
            for record in closure_records:
                for role, count in _role_counts(record["role_counts"]).items():
                    role_counts[role] = role_counts.get(role, 0) + count
            candidates.append(
                {
                    **schedule,
                    "plan": candidate_plan,
                    "execution_operator_config": candidate_plan[
                        "execution_operator_config"
                    ],
                    "execution": {
                        "forward_completed": True,
                        "backward_completed": True,
                    },
                    "required_signature_identities": required_signature_identities,
                    "bmm_call_count": len(closure_records),
                    "bmm_role_counts": {
                        role: role_counts[role] for role in sorted(role_counts)
                    },
                }
            )
            signature_occurrences.extend(closure_records)
    finally:
        _adapters.build_hd_block_gemm_plan = original_adapter_build
        _hd_plan.build_hd_block_gemm_plan = original_build
    return candidates, signature_occurrences


def _resolve_candidate_grid(args: argparse.Namespace) -> tuple[tuple[int, ...], tuple[int | str, ...]]:
    """Resolve either the full candidate grid or the explicit legacy singleton."""
    token_block = args.token_block
    feature_wave_blocks = args.feature_wave_blocks
    if (token_block is None) != (feature_wave_blocks is None):
        raise SystemExit(
            "--token-block and --feature-wave-blocks must be supplied together"
        )
    if token_block is not None:
        if token_block <= 0 or feature_wave_blocks is None or feature_wave_blocks <= 0:
            raise SystemExit("inventory schedule values must be positive")
        return (token_block,), (feature_wave_blocks,)
    return tuple(args.token_blocks), tuple(args.wave_options)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-heads", type=int, default=14,
                        help="query head count for the D=64 fixture (default: 14)")
    parser.add_argument("--key-value-heads", type=int, default=2,
                        help="KV head count; must divide query heads (default: 2)")
    parser.add_argument(
        "--sequence-lengths",
        type=_parse_sequence_lengths,
        default=_DEFAULT_SEQUENCE_LENGTHS,
    )
    parser.add_argument(
        "--token-blocks",
        type=_parse_positive_ints,
        default=_DEFAULT_TOKEN_BLOCKS,
        help="comma-separated candidate token blocks (default: 64,128)",
    )
    parser.add_argument(
        "--wave-options",
        type=_parse_wave_options,
        default=_DEFAULT_WAVE_OPTIONS,
        help="comma-separated wave sizes, with nb for all blocks or runtime:N for min(N, blocks) (default: 4,8,16,nb)",
    )
    # Kept for a focused, explicitly requested single-candidate inventory.
    # Supplying either option requires the pair so no partial grid is hidden.
    parser.add_argument("--token-block", type=int)
    parser.add_argument("--feature-wave-blocks", type=int)
    parser.add_argument("--operator-config", type=Path)
    parser.add_argument(
        "--kv-cross-impl",
        choices=("split", "batched_dense"),
        default="split",
    )
    parser.add_argument(
        "--include-performance-base",
        action="store_true",
        help="retired OP-4 option; requesting it raises an error",
    )
    parser.add_argument(
        "--feature-padding-final-parent",
        action="store_true",
        help=(
            "inventory F2160 and its unpadded control on the retained Task10 "
            "execution stack"
        ),
    )
    args = parser.parse_args(argv)
    try:
        _validate_head_geometry(args.query_heads, args.key_value_heads)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    operator_config = (
        _parse_operator_config(args.operator_config)
        if args.operator_config is not None
        else dict(_GENERIC_OPERATOR_CONFIG)
    )
    if args.feature_padding_final_parent and operator_config.get("feature_padding") != "f2160":
        raise SystemExit(
            "--feature-padding-final-parent requires an F2160 operator config"
        )
    configured_kv_cross = operator_config.get("kv_cross_impl")
    if (
        configured_kv_cross is not None
        and args.kv_cross_impl != configured_kv_cross
    ):
        raise SystemExit(
            "--kv-cross-impl must match operator config kv_cross_impl"
        )
    if args.include_performance_base:
        raise SystemExit("--include-performance-base: OP-4 exploration is retired")
    device = _canonical_device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("HD cuBLAS signature inventory requires an available CUDA device")
    token_blocks, wave_options = _resolve_candidate_grid(args)
    candidates, signatures = _run_inventory(
        device=device,
        query_heads=args.query_heads,
        key_value_heads=args.key_value_heads,
        sequence_lengths=args.sequence_lengths,
        token_blocks=token_blocks,
        wave_options=wave_options,
        operator_config=operator_config,
        include_performance_base=args.include_performance_base,
        include_kv_cross_base=args.kv_cross_impl == "batched_dense",
        include_feature_padding_base=(
            operator_config.get("feature_padding") == "f2160"
        ),
        feature_padding_final_parent=args.feature_padding_final_parent,
        kv_cross_impl=args.kv_cross_impl,
    )
    payload = _make_inventory_payload(
        device=_device_payload(device),
        environment={"torch_version": torch.__version__, "cuda_version": torch.version.cuda},
        fixture=_fixture_payload(
            query_heads=args.query_heads, key_value_heads=args.key_value_heads,
        ),
        candidate_grid={
            "sequence_lengths": list(args.sequence_lengths),
            "token_blocks": list(token_blocks),
            "wave_options": list(wave_options),
        },
        candidates=candidates,
        signatures=signatures,
        source_hashes=_source_hashes(),
        operator_config=operator_config,
    )
    _write_complete_inventory(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
