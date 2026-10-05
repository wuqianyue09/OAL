#!/usr/bin/env python3
"""Fresh-process witness for the private HD BF16 vendor-BMM contract."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import torch


_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from oal_attention.hd_cublas_compat import _canonical_device
from oal_attention.hd_block_gemm_contracts import (
    canonicalize_hd_operator_config,
)


def canonicalize_query_operator_config(config: object) -> dict[str, object]:
    """Preserve independent key and physical selectors in identities."""
    return canonicalize_hd_operator_config(config)


_SCHEMA_VERSION = "hd_bf16_bmm_probe_v1"
_MAX_NORMALIZED_ERROR = 5e-3
_MAX_POLICY_DIFFERENCE = 1e-4
_KERNEL_MARKERS = ("bf16", "gemm", "cublas", "cutlass", "xmma", "mma")
_CUDA_KERNEL_CATEGORIES = frozenset(("cuda_kernel", "gpu_kernel", "kernel"))
_DISPATCHER_EVENT_PREFIXES = ("hd_cublas_compat::",)
_COMPAT_CHILD_TIMEOUT_SECONDS = 600
_INVENTORY_EXECUTION_MODE = "inventory_fp32_diagnostic_single_current_stream_v1"
_DIAGNOSTIC_SIGNATURE_FIELDS = frozenset(
    (
        "device_capability_class",
        "execution_mode_identity",
        "provider_backend_identity",
    )
)
_SIGNATURE_TRANSFORM_SCHEMA = "hd_cublas_inventory_to_formal_signature_v1"
_INVENTORY_SCHEMA = "hd_cublas_signature_inventory_v4"
_LEGACY_INVENTORY_SCHEMAS = frozenset(("hd_cublas_signature_inventory_v3", "hd_cublas_signature_inventory_v2"))
_CANDIDATE_SCHEDULE_SCHEMA = "hd_cublas_candidate_schedule_v1"
_IDENTITY_POLICIES = ("semantic_compat", "legacy_strict")
_GENERIC_OPERATOR_CONFIG = {
    "schema_version": "hd_query_operator_config_v4",
    "query_feature_impl": "generic_materialized",
    "query_fold_impl": "generic_materialized",
    "query_gradient_flow": "materialized",
    "query_producer_fold_strategy": "none",
    "query_consumer_stages": [],
}



def _is_missing_out_dtype_api(error: TypeError) -> bool:
    message = str(error)
    return "unexpected keyword argument" in message and "out_dtype" in message


def _stable_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _evidence_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _signature_payload_identity(payload: Mapping[str, object]) -> str:
    return _evidence_sha256(payload)


def _bind_inventory_signature_to_formal(
    inventory_record: Mapping[str, object],
    *,
    formal_signature: Mapping[str, object],
    admitted_signature_key: str,
) -> dict[str, object]:
    """Allow only the declared diagnostic-to-formal signature transform."""
    inventory_identity = inventory_record.get("signature_identity_sha256")
    inventory_signature = inventory_record.get("signature")
    if not isinstance(inventory_identity, str) or not isinstance(
        inventory_signature, Mapping
    ):
        raise ValueError("inventory signature record is invalid")
    inventory_payload = dict(inventory_signature)
    formal_payload = dict(formal_signature)
    if inventory_identity != _signature_payload_identity(inventory_payload):
        raise ValueError("inventory signature identity is invalid")
    if inventory_payload.get("execution_mode_identity") != _INVENTORY_EXECUTION_MODE:
        raise ValueError("inventory signature has an unsupported diagnostic execution mode")
    inventory_structural = {
        key: value
        for key, value in inventory_payload.items()
        if key not in _DIAGNOSTIC_SIGNATURE_FIELDS
    }
    formal_structural = {
        key: value
        for key, value in formal_payload.items()
        if key not in _DIAGNOSTIC_SIGNATURE_FIELDS
    }
    if inventory_structural != formal_structural:
        raise ValueError(
            "formal signature structural payload does not match the inventory"
        )
    formal_identity = _signature_payload_identity(formal_payload)
    if admitted_signature_key != formal_identity:
        raise ValueError("admitted signature key does not bind the formal signature")
    return {
        "inventory_signature_identity_sha256": inventory_identity,
        "signature_identity_sha256": formal_identity,
        "admitted_signature_key": admitted_signature_key,
        "formal_signature": formal_payload,
        "signature_transform": {
            "schema_version": _SIGNATURE_TRANSFORM_SCHEMA,
            "diagnostic_only_fields": sorted(_DIAGNOSTIC_SIGNATURE_FIELDS),
            "inventory_execution_mode_identity": inventory_payload[
                "execution_mode_identity"
            ],
            "formal_execution_mode_identity": formal_payload.get(
                "execution_mode_identity"
            ),
            "structural_signature_identity_sha256": _signature_payload_identity(
                inventory_structural
            ),
        },
    }


@contextmanager
def _temporary_attribute(
    owner: object,
    name: str,
    value: object,
) -> Iterator[None]:
    """Temporarily change one probe-only policy and always restore it."""
    original = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, original)


def _kernel_witness_is_present(kernel_names: Sequence[str]) -> bool:
    return any(
        any(marker in name.lower() for marker in _KERNEL_MARKERS)
        for name in kernel_names
    )


def _classify_probe(
    cases: Sequence[Mapping[str, object]],
    kernel_names: Sequence[str],
) -> str:
    if not cases or not _kernel_witness_is_present(kernel_names):
        return "requires_custom_wrapper"
    for case in cases:
        policies = case.get("policies")
        if not isinstance(policies, Mapping):
            return "requires_custom_wrapper"
        for policy_name in ("enabled", "disabled"):
            result = policies.get(policy_name)
            if not isinstance(result, Mapping):
                return "requires_custom_wrapper"
            if result.get("output_dtype") != "float32":
                return "requires_custom_wrapper"
            for error_name in ("max_abs_error", "max_rel_error"):
                error = result.get(error_name)
                if not isinstance(error, (int, float)):
                    return "requires_custom_wrapper"
                if float(error) > _MAX_NORMALIZED_ERROR:
                    return "requires_custom_wrapper"
        policy_difference = case.get("policy_outputs_max_abs_difference")
        if not isinstance(policy_difference, (int, float)):
            return "requires_custom_wrapper"
        if float(policy_difference) > _MAX_POLICY_DIFFERENCE:
            return "requires_custom_wrapper"
    return "verified_vendor"


def _make_payload(
    *,
    device: Mapping[str, object],
    environment: Mapping[str, object],
    cases: Sequence[Mapping[str, object]],
    kernel_names: Sequence[str],
    source_sha256: str,
    failure: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "status": (
            "requires_custom_wrapper"
            if failure is not None
            else _classify_probe(cases, kernel_names)
        ),
        "device": dict(device),
        "environment": dict(environment),
        "cases": list(cases),
        "kernel_names": sorted(set(kernel_names)),
        "probe_source_sha256": source_sha256,
        "decision_thresholds": {
            "max_normalized_error": _MAX_NORMALIZED_ERROR,
            "max_policy_difference": _MAX_POLICY_DIFFERENCE,
            "kernel_markers": list(_KERNEL_MARKERS),
        },
    }
    if failure is not None:
        payload["failure"] = dict(failure)
    payload["identity_sha256"] = _evidence_sha256(payload)
    return payload


def _validate_compat_arguments(
    manifest: Path | None,
    inventory: Path | None,
) -> None:
    if (manifest is None) != (inventory is None):
        raise SystemExit(
            "--compat-manifest and --signature-inventory must be supplied together"
        )


def _classify_compat_probe(signature_results: Sequence[Mapping[str, object]]) -> str:
    """Keep numerical/compute and physical evidence as separate admissions."""
    if not signature_results:
        return "requires_custom_wrapper"
    for result in signature_results:
        if (
            result.get("oracle_passed") is not True
            or result.get("compute_contract_passed") is not True
        ):
            return "requires_custom_wrapper"
    if all(
        isinstance(result.get("physical_witness"), Mapping)
        and _kernel_witness_is_present(
            result["physical_witness"].get("kernel_names", ())  # type: ignore[index]
        )
        for result in signature_results
    ):
        return "verified_cublas_compat"
    return "verified_compute_contract_only"


def _signature_table_identity(
    signature_results: Sequence[Mapping[str, object]],
) -> str:
    return _evidence_sha256(
        {
            "signature_results": [
                dict(result)
                for result in sorted(
                    signature_results,
                    key=lambda result: str(result.get("signature_identity_sha256", "")),
                )
            ]
        }
    )


def _signature_is_admitted(result: Mapping[str, object]) -> bool:
    witness = result.get("physical_witness")
    return (
        result.get("oracle_passed") is True
        and result.get("compute_contract_passed") is True
        and isinstance(witness, Mapping)
        and _kernel_witness_is_present(witness.get("kernel_names", ()))
    )


def _candidate_probe_results(
    candidates: Sequence[Mapping[str, object]],
    signature_results: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Bind each physical candidate closure to its own formal witnesses."""
    result_by_inventory_id: dict[str, Mapping[str, object]] = {}
    for result in signature_results:
        inventory_id = result.get("inventory_signature_identity_sha256")
        if not isinstance(inventory_id, str) or inventory_id in result_by_inventory_id:
            raise ValueError("probe results do not uniquely bind inventory signatures")
        result_by_inventory_id[inventory_id] = result
    candidate_results: list[dict[str, object]] = []
    for candidate in candidates:
        candidate_id = candidate.get("candidate_identity_sha256")
        required = candidate.get("required_signature_identities")
        if not isinstance(candidate_id, str) or not isinstance(required, list):
            raise ValueError("inventory candidate closure is incomplete")
        results = [result_by_inventory_id.get(signature_id) for signature_id in required]
        if any(result is None for result in results):
            raise ValueError("probe results do not cover a candidate signature closure")
        bound_results = [result for result in results if result is not None]
        admitted = [
            str(result["admitted_signature_key"])
            for result in bound_results
            if _signature_is_admitted(result)
        ]
        candidate_results.append(
            {
                "candidate_identity_sha256": candidate_id,
                **(
                    {"candidate_key": dict(candidate["candidate_key"])}
                    if isinstance(candidate.get("candidate_key"), Mapping)
                    else {}
                ),
                "required_signature_identities": list(required),
                "bmm_call_count": candidate["bmm_call_count"],
                "distinct_bmm_signature_count": candidate[
                    "distinct_bmm_signature_count"
                ],
                "bmm_role_counts": dict(candidate["bmm_role_counts"]),
                "admitted_signature_keys": admitted,
                "signature_witnesses": [
                    {
                        "inventory_signature_identity_sha256": str(
                            result["inventory_signature_identity_sha256"]
                        ),
                        "physical_witness": result.get("physical_witness"),
                    }
                    for result in bound_results
                ],
                "status": "admitted"
                if len(admitted) == len(required)
                else "rejected",
            }
        )
    return sorted(
        candidate_results,
        key=lambda result: str(result["candidate_identity_sha256"]),
    )


def _make_compat_payload(
    *,
    device: Mapping[str, object],
    environment: Mapping[str, object],
    source_sha256: str,
    artifact_identity_sha256: str,
    artifact_manifest_identity_sha256: str,
    inventory_identity_sha256: str,
    provider_identity: Mapping[str, object],
    operator_metadata: Mapping[str, object],
    signature_results: Sequence[Mapping[str, object]],
    source_hashes: Mapping[str, str],
    device_capability_class: str,
    candidate_results: Sequence[Mapping[str, object]] | None = None,
    compatibility_binding: Mapping[str, object] | None = None,
    execution_binary_binding: Mapping[str, object] | None = None,
) -> dict[str, object]:
    canonical_results = [
        dict(result)
        for result in sorted(
            signature_results,
            key=lambda result: str(result.get("signature_identity_sha256", "")),
        )
    ]
    table_identity = _signature_table_identity(canonical_results)
    payload: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "status": _classify_compat_probe(canonical_results),
        "device": dict(device),
        "environment": dict(environment),
        "probe_source_sha256": source_sha256,
        "artifact_identity_sha256": artifact_identity_sha256,
        "artifact_manifest_identity_sha256": artifact_manifest_identity_sha256,
        "inventory_identity_sha256": inventory_identity_sha256,
        "provider_identity": dict(provider_identity),
        "operator_metadata": dict(operator_metadata),
        "source_hashes": dict(source_hashes),
        "signature_results": canonical_results,
        "signature_table_identity_sha256": table_identity,
        # The current loader consumes this compact frozen table; the complete
        # records above remain the source-bound audit trail.
        "admitted_signature_keys": [
            str(result["admitted_signature_key"])
            for result in canonical_results
            if _signature_is_admitted(result)
        ],
        "capability_table_identity_sha256": table_identity,
        # This value comes directly from the loaded probe-only token whose
        # keys were just generated; recomputing a lookalike risks divergence.
        "device_capability_class": device_capability_class,
        "decision_thresholds": {
            "max_normalized_error": _MAX_NORMALIZED_ERROR,
            "kernel_markers": list(_KERNEL_MARKERS),
        },
    }
    if candidate_results is not None:
        canonical_candidate_results = [
            dict(result)
            for result in sorted(
                candidate_results,
                key=lambda result: str(result.get("candidate_identity_sha256", "")),
            )
        ]
        payload["candidate_results"] = canonical_candidate_results
    if compatibility_binding is not None:
        payload["compatibility_binding"] = dict(compatibility_binding)
    if execution_binary_binding is not None:
        payload["execution_binary_binding"] = dict(execution_binary_binding)
    payload["identity_sha256"] = _evidence_sha256(payload)
    return payload


def _load_json(path: Path, *, description: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {description}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return payload


def _inventory_candidate_identity(
    *,
    sequence_length: int,
    token_block: int,
    feature_wave_blocks: int,
    operator_config: Mapping[str, object] | None = None,
) -> str:
    payload: dict[str, object] = {
            "schema": _CANDIDATE_SCHEDULE_SCHEMA,
            "sequence_length": sequence_length,
            "token_block": token_block,
            "feature_wave_blocks": feature_wave_blocks,
            "precision": "bf16_tensorcore",
            "result_contract": "full_aux",
        "local_mode": "generic_packed",
    }
    if operator_config is not None:
        payload["operator_config"] = dict(operator_config)
    return _evidence_sha256(payload)


def _new_inventory_operator_config(
    inventory: Mapping[str, object],
) -> Mapping[str, object] | None:
    """Read v4 structural selectors, allowing omissions only in legacy input."""
    if "operator_config" not in inventory:
        if inventory.get("schema_version") == _INVENTORY_SCHEMA:
            raise ValueError("HD cuBLAS inventory operator config is missing")
        return None
    config = inventory.get("operator_config")
    return canonicalize_query_operator_config(config)


def _operator_configs_match(left: object, right: object) -> bool:
    """Only structurally identical selector payloads can share probe evidence."""
    try:
        return canonicalize_query_operator_config(left) == canonicalize_query_operator_config(right)
    except (TypeError, ValueError):
        return False


def _candidate_key_matches(
    candidate: Mapping[str, object],
    *,
    operator_config: Mapping[str, object],
) -> bool:
    key = candidate.get("candidate_key")
    return (
        isinstance(key, Mapping)
        and dict(key)
        == {
            "sequence_length": candidate.get("sequence_length"),
            "token_block": candidate.get("token_block"),
            "feature_wave_blocks": candidate.get("feature_wave_blocks"),
            "operator_config": dict(operator_config),
        }
    )


def _positive_int_list(value: object, *, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"HD cuBLAS candidate grid {name} is invalid")
    if any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in value):
        raise ValueError(f"HD cuBLAS candidate grid {name} is invalid")
    if len(set(value)) != len(value):
        raise ValueError(f"HD cuBLAS candidate grid {name} has duplicates")
    return list(value)


def _role_counts(value: object, *, name: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"HD cuBLAS {name} role counts are missing")
    counts: dict[str, int] = {}
    for role, count in value.items():
        if not isinstance(role, str) or not role or not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(f"HD cuBLAS {name} role counts are invalid")
        counts[role] = count
    return {role: counts[role] for role in sorted(counts)}


def _validated_inventory_candidates(
    inventory: Mapping[str, object],
    signatures: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    """Require every declared schedule to close over real F+B signatures."""
    operator_config = _new_inventory_operator_config(inventory)
    schema_version = inventory.get("schema_version")
    raw_operator_config = inventory.get("operator_config")
    structural_config = (
        operator_config if schema_version == _INVENTORY_SCHEMA else None
    )
    grid = inventory.get("candidate_grid")
    if not isinstance(grid, Mapping) or inventory.get("candidate_grid_identity_sha256") != _evidence_sha256(dict(grid)):
        raise ValueError("HD cuBLAS candidate grid is not identity-bound")
    sequence_lengths = _positive_int_list(grid.get("sequence_lengths"), name="sequence_lengths")
    token_blocks = _positive_int_list(grid.get("token_blocks"), name="token_blocks")
    raw_wave_options = grid.get("wave_options")
    if not isinstance(raw_wave_options, list) or not raw_wave_options:
        raise ValueError("HD cuBLAS candidate grid wave_options is invalid")
    wave_options: list[int | str] = []
    for option in raw_wave_options:
        if option == "number_blocks":
            normalized: int | str = option
        elif isinstance(option, str) and re.fullmatch(r"runtime:[1-9][0-9]*", option):
            normalized = option
        elif isinstance(option, int) and not isinstance(option, bool) and option > 0:
            normalized = option
        else:
            raise ValueError("HD cuBLAS candidate grid wave_options is invalid")
        if normalized in wave_options:
            raise ValueError("HD cuBLAS candidate grid wave_options has duplicates")
        wave_options.append(normalized)

    expected_ids: set[str] = set()
    expected_number_blocks: dict[str, int] = {}
    for sequence_length in sequence_lengths:
        for token_block in token_blocks:
            number_blocks = (sequence_length + token_block - 1) // token_block
            for option in wave_options:
                wave = number_blocks if option == "number_blocks" else option
                if isinstance(option, str) and option.startswith("runtime:"):
                    wave = min(number_blocks, int(option.split(":", 1)[1]))
                assert isinstance(wave, int)
                if wave > number_blocks:
                    continue
                candidate_id = _inventory_candidate_identity(
                    sequence_length=sequence_length,
                    token_block=token_block,
                    feature_wave_blocks=wave,
                    operator_config=structural_config,
                )
                expected_ids.add(candidate_id)
                expected_number_blocks[candidate_id] = number_blocks
    candidates = inventory.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("HD cuBLAS signature inventory has no candidate closures")
    if candidates != sorted(
        candidates,
        key=lambda candidate: str(candidate.get("candidate_identity_sha256", ""))
        if isinstance(candidate, Mapping)
        else "",
    ):
        raise ValueError("HD cuBLAS candidate closures are not canonical")
    signature_ids = {
        str(record["signature_identity_sha256"]) for record in signatures
    }
    observed_ids: set[str] = set()
    aggregate_candidate_roles: dict[str, int] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("HD cuBLAS candidate closure is invalid")
        candidate_id = candidate.get("candidate_identity_sha256")
        sequence_length = candidate.get("sequence_length")
        token_block = candidate.get("token_block")
        wave = candidate.get("feature_wave_blocks")
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in (sequence_length, token_block, wave)
        ) or not isinstance(candidate_id, str):
            raise ValueError("HD cuBLAS candidate closure is incomplete")
        if candidate_id != _inventory_candidate_identity(
            sequence_length=sequence_length,
            token_block=token_block,
            feature_wave_blocks=wave,
            operator_config=structural_config,
        ) or candidate_id in observed_ids or candidate_id not in expected_ids:
            raise ValueError("HD cuBLAS candidate identity is invalid")
        if structural_config is not None and not _candidate_key_matches(
            candidate,
            operator_config=structural_config,
        ):
            raise ValueError("HD cuBLAS candidate key is invalid")
        plan = candidate.get("plan")
        if structural_config is not None and (
            candidate.get("execution_operator_config") != structural_config
            or not isinstance(plan, Mapping)
            or plan.get("execution_operator_config") != structural_config
        ):
            raise ValueError("HD cuBLAS candidate execution operator config is invalid")
        if (
            schema_version == "hd_cublas_signature_inventory_v3"
            and isinstance(raw_operator_config, Mapping)
            and not _candidate_key_matches(
                candidate,
                operator_config=raw_operator_config,
            )
        ):
            raise ValueError("HD cuBLAS candidate key is invalid")
        observed_ids.add(candidate_id)
        execution = candidate.get("execution")
        required = candidate.get("required_signature_identities")
        bmm_call_count = candidate.get("bmm_call_count")
        role_counts = _role_counts(candidate.get("bmm_role_counts"), name="candidate")
        if (
            not isinstance(plan, Mapping)
            or not isinstance(plan.get("plan_id"), str)
            or plan.get("result_contract") != "full_aux"
            or plan.get("number_blocks") != expected_number_blocks[candidate_id]
            or not isinstance(execution, Mapping)
            or execution.get("forward_completed") is not True
            or execution.get("backward_completed") is not True
            or not isinstance(required, list)
            or not required
            or not all(isinstance(signature_id, str) for signature_id in required)
            or required != sorted(required)
            or len(set(required)) != len(required)
            or not set(required) <= signature_ids
            or candidate.get("distinct_bmm_signature_count") != len(required)
            or not isinstance(bmm_call_count, int)
            or isinstance(bmm_call_count, bool)
            or bmm_call_count <= 0
            or sum(role_counts.values()) != bmm_call_count
        ):
            raise ValueError("HD cuBLAS candidate signature closure is incomplete")
        for role, count in role_counts.items():
            aggregate_candidate_roles[role] = aggregate_candidate_roles.get(role, 0) + count
    if observed_ids != expected_ids:
        raise ValueError("HD cuBLAS candidate grid is not fully inventoried")

    aggregate_signature_roles: dict[str, int] = {}
    for record in signatures:
        for role, count in _role_counts(record.get("role_counts"), name="signature").items():
            aggregate_signature_roles[role] = aggregate_signature_roles.get(role, 0) + count
    if aggregate_signature_roles != aggregate_candidate_roles:
        raise ValueError("HD cuBLAS candidate role accounting does not close")
    return list(candidates)


def _validate_inventory_payload(
    inventory: Mapping[str, object],
    *,
    identity_policy: str = "semantic_compat",
) -> list[Mapping[str, object]]:
    if identity_policy not in _IDENTITY_POLICIES:
        raise ValueError("unsupported identity policy")
    if inventory.get("schema_version") not in (
        _INVENTORY_SCHEMA,
        *_LEGACY_INVENTORY_SCHEMAS,
    ):
        raise ValueError("HD cuBLAS signature inventory schema is unsupported")
    identity = inventory.get("identity_sha256")
    if identity_policy == "legacy_strict" and (
        not isinstance(identity, str)
        or identity
        != _evidence_sha256(
            {key: value for key, value in inventory.items() if key != "identity_sha256"}
        )
    ):
        raise ValueError("HD cuBLAS signature inventory identity is invalid")
    execution = inventory.get("execution")
    if not isinstance(execution, Mapping) or not (
        execution.get("forward_completed") is True
        and execution.get("backward_completed") is True
    ):
        raise ValueError("HD cuBLAS signature inventory lacks full forward+backward execution")
    signatures = inventory.get("signatures")
    if not isinstance(signatures, list) or not signatures:
        raise ValueError("HD cuBLAS signature inventory is empty")
    if signatures != sorted(
        signatures,
        key=lambda record: str(record.get("signature_identity_sha256", ""))
        if isinstance(record, Mapping)
        else "",
    ):
        raise ValueError("HD cuBLAS signature inventory is not canonical")
    if any(
        not isinstance(record, Mapping)
        or not isinstance(record.get("signature_identity_sha256"), str)
        or not isinstance(record.get("signature"), Mapping)
        for record in signatures
    ):
        raise ValueError("HD cuBLAS signature inventory records are invalid")
    _validated_inventory_candidates(inventory, signatures)
    return list(signatures)


def _view_for_inventory_signature(
    signature: Mapping[str, object],
    *,
    operand: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = signature.get("shape")
    layouts = signature.get("layouts")
    offsets = signature.get("storage_offsets")
    pointer_modulo = signature.get("pointer_mod_256")
    batch_strides = signature.get("batch_strides")
    leading_dimensions = signature.get("leading_dimensions")
    ops = signature.get("ops")
    if not (
        isinstance(shape, list)
        and len(shape) == 4
        and isinstance(layouts, list)
        and len(layouts) == 3
        and isinstance(offsets, list)
        and len(offsets) == 3
        and isinstance(pointer_modulo, list)
        and len(pointer_modulo) == 3
        and isinstance(batch_strides, list)
        and len(batch_strides) == 3
        and isinstance(leading_dimensions, list)
        and len(leading_dimensions) == 3
        and isinstance(ops, list)
        and len(ops) == 2
    ):
        raise ValueError("inventory signature layout metadata is incomplete")
    batch, rows, columns, reduction = (int(value) for value in shape)
    if operand == 0:
        tensor_rows, tensor_columns, dtype, layout = rows, reduction, torch.bfloat16, layouts[0]
    elif operand == 1:
        tensor_rows, tensor_columns, dtype, layout = reduction, columns, torch.bfloat16, layouts[1]
    elif operand == 2:
        tensor_rows, tensor_columns, dtype, layout = rows, columns, torch.float32, layouts[2]
    else:  # pragma: no cover - private caller owns this invariant.
        raise ValueError("inventory operand is invalid")
    if layout == "R":
        strides = (tensor_rows * tensor_columns, tensor_columns, 1)
    elif layout == "T" and operand != 2:
        strides = (tensor_rows * tensor_columns, 1, tensor_rows)
    else:
        raise ValueError("inventory signature uses unsupported dense layout")
    # Layouts are in logical ``left, right, out`` order while the physical
    # cuBLAS metadata is ``right, left, out`` to match GEMM operand order.
    metadata_index = (1, 0, 2)[operand]
    expected_ld = tensor_columns if layout == "R" else tensor_rows
    if int(batch_strides[metadata_index]) != strides[0]:
        raise ValueError("inventory signature batch stride does not match its layout")
    if int(leading_dimensions[metadata_index]) != expected_ld:
        raise ValueError("inventory signature leading dimension does not match its layout")
    if operand != 2 and ops[metadata_index] != ("N" if layout == "R" else "T"):
        raise ValueError("inventory signature transpose operation does not match its layout")
    offset = int(offsets[metadata_index])
    base = torch.empty(
        offset + batch * strides[0], dtype=dtype, device=device
    )
    view = torch.as_strided(
        base,
        size=(batch, tensor_rows, tensor_columns),
        stride=strides,
        storage_offset=offset,
    )
    if int(view.data_ptr()) % 256 != int(pointer_modulo[metadata_index]):
        raise RuntimeError("could not rebuild the inventory pointer alignment")
    return base, view


def _fill_probe_operands(
    left: torch.Tensor,
    right: torch.Tensor,
    out: torch.Tensor,
    *,
    signature_identity: str,
) -> None:
    seed = int(signature_identity[:16], 16)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    left.copy_(
        torch.randn(left.shape, generator=generator, dtype=torch.float32)
        .to(device=left.device, dtype=left.dtype)
    )
    right.copy_(
        torch.randn(right.shape, generator=generator, dtype=torch.float32)
        .to(device=right.device, dtype=right.dtype)
    )
    out.zero_()


def _is_dispatcher_event_name(name: str) -> bool:
    return any(name.lower().startswith(prefix) for prefix in _DISPATCHER_EVENT_PREFIXES)


def _profiler_event_category(event: object) -> str:
    """Extract Kineto's kernel category without trusting the display name."""
    for attribute in ("category", "cat"):
        value = getattr(event, attribute, None)
        if isinstance(value, str) and value:
            return value.lower()
    metadata = getattr(event, "metadata_json", None)
    if isinstance(metadata, str) and metadata:
        try:
            decoded = json.loads(metadata)
        except json.JSONDecodeError:
            return ""
        if isinstance(decoded, Mapping):
            for field in ("category", "cat"):
                value = decoded.get(field)
                if isinstance(value, str) and value:
                    return value.lower()
    return ""


def _is_direct_cuda_kernel_event(event: object) -> bool:
    name = str(getattr(event, "name", ""))
    if not name or _is_dispatcher_event_name(name):
        return False
    device_type = str(getattr(event, "device_type", "")).lower()
    category = _profiler_event_category(event)
    return (
        "cuda" in device_type
        and category in _CUDA_KERNEL_CATEGORIES
        and float(getattr(event, "self_cuda_time_total", 0.0)) > 0.0
    )


def _attached_cuda_kernel_names(event: object) -> list[str]:
    """Read profiler-owned raw kernel records attached to a host op."""
    names: list[str] = []
    kernels = getattr(event, "kernels", ())
    if not isinstance(kernels, Sequence):
        return names
    for kernel in kernels:
        name = str(getattr(kernel, "name", ""))
        device = str(getattr(kernel, "device", "")).lower()
        duration = getattr(kernel, "duration", 0.0)
        if (
            name
            and not _is_dispatcher_event_name(name)
            and "cuda" in device
            and isinstance(duration, (int, float))
            and float(duration) > 0.0
        ):
            names.append(name)
    return names


def _raw_cuda_kernel_names(events: Sequence[object]) -> list[str]:
    """Return profiler-designated CUDA kernels, never operator/dispatcher labels."""
    names: list[str] = []
    for event in events:
        if _is_direct_cuda_kernel_event(event):
            names.append(str(getattr(event, "name", "")))
        names.extend(_attached_cuda_kernel_names(event))
    return sorted(set(names))


def _raw_cuda_kernel_names_from_chrome_trace(
    trace: Mapping[str, object],
) -> list[str]:
    """Extract raw CUDA kernel records that Kineto could not attach to a host op."""
    events = trace.get("traceEvents")
    if not isinstance(events, list):
        return []
    names: list[str] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        name = str(event.get("name", ""))
        category = str(event.get("cat", "")).lower()
        if (
            name
            and not _is_dispatcher_event_name(name)
            and category in _CUDA_KERNEL_CATEGORIES
        ):
            names.append(name)
    return sorted(set(names))


def _compat_kernel_names(
    operator: object,
    left: torch.Tensor,
    right: torch.Tensor,
    out: torch.Tensor,
) -> list[str]:
    from torch.profiler import ProfilerActivity, profile

    if not callable(operator):
        raise TypeError("loaded HD cuBLAS operator is not callable")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        operator(left, right, out=out)
        torch.cuda.synchronize(left.device)
    with tempfile.TemporaryDirectory(prefix="hd_cublas_compat_probe_") as directory:
        trace_path = Path(directory) / "kineto_trace.json"
        prof.export_chrome_trace(str(trace_path))
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(trace, Mapping):
        raise RuntimeError("Kineto exported an invalid Chrome trace")
    return sorted(
        set(
            _raw_cuda_kernel_names(prof.events())
            + _raw_cuda_kernel_names_from_chrome_trace(trace)
        )
    )


def _physical_witness_payload(kernel_names: Sequence[str]) -> dict[str, object]:
    raw_kernel_names = sorted(set(kernel_names))
    return {
        "raw_cuda_kernel_names": raw_kernel_names,
        "physical_witness": (
            {"kernel_names": raw_kernel_names}
            if _kernel_witness_is_present(raw_kernel_names)
            else None
        ),
    }


def _probe_compat_signature(
    record: Mapping[str, object],
    *,
    token: object,
    device: torch.device,
) -> dict[str, object]:
    from oal_attention.hd_cublas_compat import (
        _contraction_signature_key,
        _contraction_signature_payload,
        _contraction_runtime_signature_key,
    )

    signature_identity = record.get("signature_identity_sha256")
    signature = record.get("signature")
    if not isinstance(signature_identity, str) or not isinstance(signature, Mapping):
        raise ValueError("inventory signature record is invalid")
    retained, left = _view_for_inventory_signature(signature, operand=0, device=device)
    retained_right, right = _view_for_inventory_signature(signature, operand=1, device=device)
    retained_out, out = _view_for_inventory_signature(signature, operand=2, device=device)
    keepalive = (retained, retained_right, retained_out)
    del keepalive
    _fill_probe_operands(
        left,
        right,
        out,
        signature_identity=signature_identity,
    )
    operator = getattr(token, "operator")
    identity = getattr(token, "identity")
    execution_mode_identity = getattr(token, "execution_mode_identity")
    formal_signature = _contraction_signature_payload(
        left,
        right,
        out,
        backend_identity=identity,
        execution_mode_identity=execution_mode_identity,
    )
    admitted_signature_key = _contraction_signature_key(
        left,
        right,
        out,
        backend_identity=identity,
        execution_mode_identity=execution_mode_identity,
    )
    runtime_signature_key = _contraction_runtime_signature_key(left, right, out)
    binding = _bind_inventory_signature_to_formal(
        record,
        formal_signature=formal_signature,
        admitted_signature_key=admitted_signature_key,
    )
    reference = torch.bmm(left.float(), right.float())
    output_pointer = int(out.data_ptr())
    operator(left, right, out=out)
    torch.cuda.synchronize(device)
    error = _errors(out, reference)
    kernel_names = _compat_kernel_names(operator, left, right, out)
    return {
        **binding,
        "runtime_signature_key": list(runtime_signature_key),
        "oracle_passed": (
            error["max_abs_error"] <= _MAX_NORMALIZED_ERROR
            and error["max_rel_error"] <= _MAX_NORMALIZED_ERROR
        ),
        "compute_contract_passed": (
            left.dtype == torch.bfloat16
            and right.dtype == torch.bfloat16
            and out.dtype == torch.float32
            and int(out.data_ptr()) == output_pointer
        ),
        "errors": error,
        **_physical_witness_payload(kernel_names),
    }


def _run_compat_probe_locally(
    *,
    manifest_path: Path,
    inventory_path: Path,
    device: torch.device,
    source_sha256: str,
) -> dict[str, object]:
    from oal_attention import hd_cublas_compat as compat

    manifest = _load_json(manifest_path, description="HD cuBLAS manifest")
    inventory = _load_json(inventory_path, description="HD cuBLAS signature inventory")
    records = _validate_inventory_payload(inventory)
    candidates = _validated_inventory_candidates(inventory, records)
    token = compat.load_hd_cublas_compat(manifest_path, None, device)
    provider_identity = compat._current_provider_identity(device)
    metadata = compat._LOADED_METADATA
    if not isinstance(metadata, Mapping):
        raise RuntimeError("loaded HD cuBLAS metadata is unavailable")
    manifest_identity = manifest.get("manifest_identity_sha256")
    artifact_identity = manifest.get("artifact_identity_sha256")
    inventory_identity = inventory.get("identity_sha256")
    if not all(isinstance(value, str) for value in (manifest_identity, artifact_identity, inventory_identity)):
        raise ValueError("HD cuBLAS evidence identities are missing")
    results = [
        _probe_compat_signature(record, token=token, device=device)
        for record in records
    ]
    compatibility = manifest.get("load_compatibility")
    if not isinstance(compatibility, Mapping):
        raise ValueError("HD cuBLAS manifest compatibility metadata is missing")
    compatibility_binding = {
        "artifact_identity_sha256": artifact_identity,
        "artifact_manifest_identity_sha256": manifest_identity,
        "shared_object_sha256": manifest.get("shared_object_sha256"),
        "provider_sha256": compatibility.get("provider_sha256"),
        "provider_soname": compatibility.get("provider_soname"),
        "runtime_cublas_version": compatibility.get("runtime_cublas_version"),
    }
    execution_binary_binding = {
        "artifact_identity_sha256": artifact_identity,
        "artifact_manifest_identity_sha256": manifest_identity,
        "shared_object_sha256": manifest.get("shared_object_sha256"),
        "provider_sha256": provider_identity.get("provider_sha256"),
        "provider_soname": provider_identity.get("provider_soname"),
        "runtime_cublas_version": metadata.get("runtime_cublas_version"),
    }
    return _make_compat_payload(
        device=_device_payload(device),
        environment={
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "allow_bf16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
        },
        source_sha256=source_sha256,
        artifact_identity_sha256=artifact_identity,
        artifact_manifest_identity_sha256=manifest_identity,
        inventory_identity_sha256=inventory_identity,
        provider_identity=provider_identity,
        operator_metadata=metadata,
        signature_results=results,
        source_hashes={
            "loader": hashlib.sha256(
                (_ROOT / "oal_attention" / "hd_cublas_compat.py").read_bytes()
            ).hexdigest(),
            "facade": hashlib.sha256(
                (_ROOT / "oal_attention" / "hd_block_gemm_contractions.py").read_bytes()
            ).hexdigest(),
            "operator": hashlib.sha256(
                (_ROOT / "oal_attention" / "csrc" / "hd_cublas_compat.cpp").read_bytes()
            ).hexdigest(),
        },
        device_capability_class=token.identity.device_capability_class,
        candidate_results=_candidate_probe_results(candidates, results),
        compatibility_binding=compatibility_binding,
        execution_binary_binding=execution_binary_binding,
    )


class _CompatChildResult:
    __slots__ = ("returncode", "stdout", "stderr", "failure_kind", "timeout_seconds")

    def __init__(
        self,
        *,
        returncode: int,
        stdout: str,
        stderr: str,
        failure_kind: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.failure_kind = failure_kind
        self.timeout_seconds = timeout_seconds


def _validate_compat_parent_inputs(
    *,
    manifest_path: Path,
    inventory_path: Path,
) -> None:
    """Validate static evidence before dispatching the disposable loader."""
    manifest = _load_json(manifest_path, description="HD cuBLAS manifest")
    if manifest.get("schema_version") != "hd_cublas_compat_artifact_v1":
        raise ValueError("HD cuBLAS manifest schema is unsupported")
    if not isinstance(manifest.get("artifact_identity_sha256"), str) or not isinstance(
        manifest.get("manifest_identity_sha256"), str
    ):
        raise ValueError("HD cuBLAS manifest identities are missing")
    inventory = _load_json(inventory_path, description="HD cuBLAS signature inventory")
    _validate_inventory_payload(inventory)


def _invoke_compat_probe_child(
    *,
    manifest_path: Path,
    inventory_path: Path,
    device: torch.device,
) -> _CompatChildResult:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_compat-child",
        "--device",
        str(device),
        "--output",
        os.devnull,
        "--compat-manifest",
        str(manifest_path),
        "--signature-inventory",
        str(inventory_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=_COMPAT_CHILD_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        stderr = error.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return _CompatChildResult(
            returncode=-1,
            stdout="",
            stderr=str(stderr or "compat probe child exceeded its timeout"),
            failure_kind="compat_probe_child_timed_out",
            timeout_seconds=_COMPAT_CHILD_TIMEOUT_SECONDS,
        )
    return _CompatChildResult(
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _compat_child_rejection(
    *,
    source_sha256: str,
    returncode: int,
    stderr: str,
    failure_kind: str = "compat_probe_child_failed",
    timeout_seconds: int | None = None,
) -> dict[str, object]:
    failure: dict[str, object] = {
        "kind": failure_kind,
        "returncode": returncode,
        "stderr": stderr,
    }
    if timeout_seconds is not None:
        failure["timeout_seconds"] = timeout_seconds
    payload: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "status": "requires_custom_wrapper",
        "probe_source_sha256": source_sha256,
        "failure": failure,
    }
    payload["identity_sha256"] = _evidence_sha256(payload)
    return payload


def _run_compat_probe(
    *,
    manifest_path: Path,
    inventory_path: Path,
    device: torch.device,
    source_sha256: str,
) -> dict[str, object]:
    """Run all artifact loading and operator checks in a disposable child."""
    _validate_compat_parent_inputs(
        manifest_path=manifest_path,
        inventory_path=inventory_path,
    )
    child = _invoke_compat_probe_child(
        manifest_path=manifest_path,
        inventory_path=inventory_path,
        device=device,
    )
    if child.returncode not in {0, 2}:
        return _compat_child_rejection(
            source_sha256=source_sha256,
            returncode=child.returncode,
            stderr=child.stderr,
            failure_kind=child.failure_kind or "compat_probe_child_failed",
            timeout_seconds=child.timeout_seconds,
        )
    if child.returncode == 2 and not child.stdout.strip():
        return _compat_child_rejection(
            source_sha256=source_sha256,
            returncode=child.returncode,
            stderr=child.stderr,
            failure_kind=child.failure_kind or "compat_probe_child_failed",
            timeout_seconds=child.timeout_seconds,
        )
    try:
        payload = json.loads(child.stdout)
    except json.JSONDecodeError:
        return _compat_child_rejection(
            source_sha256=source_sha256,
            returncode=child.returncode,
            stderr="compat probe child did not emit one JSON payload",
        )
    expected_returncodes = {
        "verified_cublas_compat": 0,
        "verified_compute_contract_only": 2,
        "requires_custom_wrapper": 2,
    }
    status = payload.get("status") if isinstance(payload, dict) else None
    if not isinstance(status, str) or expected_returncodes.get(status) != child.returncode:
        return _compat_child_rejection(
            source_sha256=source_sha256,
            returncode=child.returncode,
            stderr="compat probe child returned an invalid status/exit-code pair",
        )
    identity = payload.get("identity_sha256")
    if not isinstance(identity, str) or identity != _evidence_sha256(
        {key: value for key, value in payload.items() if key != "identity_sha256"}
    ):
        return _compat_child_rejection(
            source_sha256=source_sha256,
            returncode=child.returncode,
            stderr="compat probe child identity is invalid",
        )
    return payload


def _adversarial_inputs(
    shape: tuple[int, int, int, int],
    *,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, rows, reduction, columns = shape
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    left = torch.randn(batch, rows, reduction, generator=generator)
    right = torch.randn(batch, reduction, columns, generator=generator)
    alternating = torch.ones(reduction)
    alternating[1::2] = -1.0
    left.mul_(alternating.view(1, 1, reduction))
    right.add_(alternating.view(1, reduction, 1) * (2.0**-5))
    return (
        left.to(device=device, dtype=torch.bfloat16),
        right.to(device=device, dtype=torch.bfloat16),
    )


def _errors(
    actual: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float]:
    reference_cpu = reference.double().cpu()
    difference = (actual.double().cpu() - reference_cpu).abs()
    reference_scale = max(float(reference_cpu.abs().max().item()), 1.0)
    max_abs = float(difference.max().item())
    return {
        "max_abs_error": max_abs / reference_scale,
        "max_rel_error": max_abs / reference_scale,
        "raw_max_abs_error": max_abs,
        "reference_abs_max": reference_scale,
    }


def _profile_kernel_names(
    left: torch.Tensor,
    right: torch.Tensor,
) -> list[str]:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        torch.bmm(left, right, out_dtype=torch.float32)
        torch.cuda.synchronize(left.device)
    names: list[str] = []
    for event in prof.events():
        device_type = str(getattr(event, "device_type", "")).lower()
        if "cuda" in device_type or getattr(event, "self_cuda_time_total", 0) > 0:
            names.append(str(event.name))
    return sorted(set(names))


def _run_case(
    shape: tuple[int, int, int, int],
    *,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, object], list[str]]:
    left, right = _adversarial_inputs(shape, device=device, seed=seed)
    left_cpu = left.cpu().double()
    right_cpu = right.cpu().double()
    fp64_reference = torch.bmm(left_cpu, right_cpu)
    explicit_fp32_reference = torch.sum(
        left.float().unsqueeze(-1) * right.float().unsqueeze(-3),
        dim=2,
        dtype=torch.float32,
    )
    torch.cuda.synchronize(device)

    matmul = torch.backends.cuda.matmul
    outputs: dict[str, torch.Tensor] = {}
    policies: dict[str, object] = {}
    for label, enabled in (("enabled", True), ("disabled", False)):
        with _temporary_attribute(
            matmul,
            "allow_bf16_reduced_precision_reduction",
            enabled,
        ):
            output = torch.bmm(left, right, out_dtype=torch.float32)
            torch.cuda.synchronize(device)
        outputs[label] = output
        policy_result: dict[str, object] = {
            "allow_bf16_reduced_precision_reduction": enabled,
            "output_dtype": str(output.dtype).removeprefix("torch."),
            **_errors(output, fp64_reference),
        }
        fp32_difference = (output - explicit_fp32_reference).abs()
        policy_result["explicit_fp32_oracle_max_abs_error"] = float(
            fp32_difference.max().item()
        )
        policies[label] = policy_result

    kernel_names = _profile_kernel_names(left, right)
    case = {
        "shape": list(shape),
        "seed": seed,
        "policies": policies,
        "policy_outputs_max_abs_difference": float(
            (outputs["enabled"] - outputs["disabled"]).abs().max().item()
        ),
    }
    return case, kernel_names


def _device_payload(device: torch.device) -> dict[str, object]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "type": "cuda",
        "index": index,
        "name": properties.name,
        "capability": list(torch.cuda.get_device_capability(index)),
        "total_memory": properties.total_memory,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compat-manifest", type=Path)
    parser.add_argument("--signature-inventory", type=Path)
    parser.add_argument("--_compat-child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_compat_arguments(args.compat_manifest, args.signature_inventory)
    device = _canonical_device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("BF16 BMM probe requires an available CUDA device")
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.compat_manifest is not None:
        if args._compat_child:
            payload = _run_compat_probe_locally(
                manifest_path=args.compat_manifest,
                inventory_path=args.signature_inventory,
                device=device,
                source_sha256=source_sha256,
            )
            print(json.dumps(payload, sort_keys=True))
            return 0 if payload["status"] == "verified_cublas_compat" else 2
        payload = _run_compat_probe(
            manifest_path=args.compat_manifest,
            inventory_path=args.signature_inventory,
            device=device,
            source_sha256=source_sha256,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 0 if payload["status"] == "verified_cublas_compat" else 2
    if args._compat_child:
        raise SystemExit("--_compat-child requires compatibility evidence arguments")

    cases: list[dict[str, object]] = []
    kernel_names: list[str] = []
    failure = None
    try:
        for seed, shape in enumerate(
            ((1, 64, 256, 64), (2, 37, 257, 53)),
            start=1729,
        ):
            case, witnessed_kernels = _run_case(shape, device=device, seed=seed)
            cases.append(case)
            kernel_names.extend(witnessed_kernels)
    except TypeError as error:
        if not _is_missing_out_dtype_api(error):
            raise
        failure = {
            "kind": "unsupported_torch_bmm_out_dtype_api",
            "message": str(error),
        }
    payload = _make_payload(
        device=_device_payload(device),
        environment={
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "allow_bf16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
        },
        cases=cases,
        kernel_names=kernel_names,
        source_sha256=source_sha256,
        failure=failure,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0 if payload["status"] == "verified_vendor" else 2


if __name__ == "__main__":
    raise SystemExit(main())
