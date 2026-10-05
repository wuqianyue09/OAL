"""Canonical logical-buffer compilation for private HD Block-GEMM plans."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, prod
from typing import Any, Sequence

from .hd_block_gemm_cache import PairMetadataCacheEntry
from .hd_block_gemm_contracts import (
    _sha256,
    _stable_json,
    _validate_backward_schedule,
    _validate_backward_normalize_impl,
    _validate_gradient_staging,
    _validate_key_feature_impl,
    _validate_key_fold_impl,
    _validate_query_operator_identity,
    _validate_key_retention,
    _validate_save_local_score,
)

_FEATURE_STORAGE = "float32"
_DTYPE_BYTES = {
    "bool": 1,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
    "int64": 8,
}
_EXECUTION_STAGES = (
    "forward_totals",
    "forward_carry_scan",
    "forward_feature_wave",
    "forward_outputs_saved",
    "backward_normalization",
    "backward_query_wave",
    "backward_right_scan",
    "backward_shared_wave",
    "backward_kv_wave",
)
_STAGE_INDEX = {stage: index for index, stage in enumerate(_EXECUTION_STAGES)}


@dataclass(frozen=True)
class HDLogicalBuffer:
    """One named physical buffer and its inclusive logical lifetime."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    start_stage: str
    end_stage: str
    nbytes: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("logical buffer name must be a non-empty string")
        if not isinstance(self.shape, tuple) or not self.shape:
            raise TypeError("logical buffer shape must be a non-empty tuple")
        if any(
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension <= 0
            for dimension in self.shape
        ):
            raise ValueError("logical buffer dimensions must be positive integers")
        if self.dtype not in _DTYPE_BYTES:
            raise ValueError(f"unsupported logical buffer dtype: {self.dtype}")
        if self.start_stage not in _STAGE_INDEX or self.end_stage not in _STAGE_INDEX:
            raise ValueError("logical buffer lifetime uses an unknown execution stage")
        if _STAGE_INDEX[self.start_stage] > _STAGE_INDEX[self.end_stage]:
            raise ValueError("logical buffer lifetime cannot run backwards")
        expected_nbytes = prod(self.shape) * _DTYPE_BYTES[self.dtype]
        if self.nbytes and self.nbytes != expected_nbytes:
            raise ValueError("logical buffer byte count does not match shape and dtype")
        object.__setattr__(self, "nbytes", expected_nbytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "nbytes": self.nbytes,
            "lifetime": [self.start_stage, self.end_stage],
        }


def _peak_bytes(
    buffers: tuple[HDLogicalBuffer, ...],
    stages: tuple[str, ...],
) -> int:
    if not buffers or not stages:
        return 0
    return max(
        sum(
            buffer.nbytes
            for buffer in buffers
            if _STAGE_INDEX[buffer.start_stage]
            <= _STAGE_INDEX[stage]
            <= _STAGE_INDEX[buffer.end_stage]
        )
        for stage in stages
    )


def _byte_summaries(
    buffers: tuple[HDLogicalBuffer, ...],
    gradient_mask: tuple[bool, bool, bool, bool, bool, bool],
    cache_before_manifest: tuple[PairMetadataCacheEntry, ...],
) -> dict[str, int]:
    """Derive every reported byte count from the named lifetime table."""
    by_name = {buffer.name: buffer for buffer in buffers}

    def total(*names: str) -> int:
        return sum(by_name[name].nbytes for name in names if name in by_name)

    backward_requested = any(gradient_mask)
    need_q, need_k, need_v, need_a, need_b, need_c = gradient_mask
    query_side = need_q or need_a or need_b or need_c
    forward_scratch_names = {
        "forward_phi_k_cache",
        "forward_phi_k_wave",
        "forward_phi_q_wave",
        "forward_phi_k_query_wave",
        "forward_carry_query_wave",
        "forward_u_query_wave",
        "forward_phi_q_input_work",
        "forward_phi_k_input_work",
        "forward_pair_work",
        "forward_pair_second_work",
        "forward_u_wave",
        "forward_local_score",
        "forward_local_score_tc",
        "forward_local_y",
        "forward_y_wave",
        "forward_block_totals_wave",
        "forward_scan_inclusive",
    }
    if not query_side:
        forward_scratch_names.add("block_totals_carry")
    saved_names = [
        name for name in ("saved_q", "saved_k", "saved_v") if name in by_name
    ]
    saved_coefficient_names: list[str] = []
    if need_k or need_v:
        saved_coefficient_names.append("logical_a")
    if need_q or need_k or need_v:
        saved_coefficient_names.extend(("logical_b", "logical_c"))
    if need_q or need_k or need_v or need_b or need_c:
        saved_coefficient_names.append("logical_scale")
    saved_names.extend(saved_coefficient_names)
    if backward_requested:
        saved_names.extend(("output", "numerator", "denominator"))
    if "saved_local_score_tc" in by_name:
        saved_names.append("saved_local_score_tc")
    if "saved_phi_k_tc" in by_name:
        saved_names.append("saved_phi_k_tc")
    backward_scratch = tuple(
        buffer for buffer in buffers if buffer.name.startswith("backward_")
    )
    forward_feature_names = (
        "forward_phi_k_wave",
        "forward_phi_q_wave",
        "forward_phi_k_query_wave",
        "forward_carry_query_wave",
        "forward_u_query_wave",
        "forward_u_wave",
        "forward_y_wave",
    )
    forward_scratch = tuple(
        buffer for buffer in buffers if buffer.name in forward_scratch_names
    )
    saved_tensor_bytes = total(*saved_names)
    saved_carry_bytes = total("block_totals_carry") if query_side else 0
    current_cache_names = {
        "pair_rows_cache",
        "pair_columns_cache",
        "pair_multiplicity_cache",
    }
    core_metadata_names = tuple(
        buffer.name
        for buffer in buffers
        if buffer.name in current_cache_names
        or buffer.name.startswith("pair_cache_entry_")
    )
    materialized_metadata_names = (
        "pair_rows_materialized",
        "pair_columns_materialized",
        "pair_multiplicity_materialized",
    )
    return {
        "forward_block_totals_bytes": total("block_totals_carry"),
        "saved_carry_bytes": saved_carry_bytes,
        "output_bytes": total("output", "numerator", "denominator"),
        "largest_feature_wave_bytes": total(*forward_feature_names),
        "local_score_bytes": total("forward_local_score", "forward_local_score_tc"),
        "forward_ephemeral_peak_bytes": _peak_bytes(
            forward_scratch,
            _EXECUTION_STAGES[:4],
        ),
        "saved_tensor_bytes": saved_tensor_bytes,
        "saved_coefficient_bytes": total(*saved_coefficient_names),
        "saved_activation_bytes": saved_tensor_bytes + saved_carry_bytes,
        "cache_before_residency_bytes": sum(
            entry.nbytes for entry in cache_before_manifest
        ),
        "core_metadata_residency_bytes": total(*core_metadata_names),
        "pair_metadata_materialization_bytes": total(*materialized_metadata_names),
        "backward_right_scan_bytes": total(
            "backward_dcarry",
            "backward_right_scan_dt",
            "backward_right_scan_total",
        ),
        "backward_scratch_peak_bytes": _peak_bytes(
            backward_scratch,
            _EXECUTION_STAGES[4:],
        ),
        "invocation_peak_bytes": _peak_bytes(buffers, _EXECUTION_STAGES),
    }


def _expected_logical_buffers(
    *,
    geometry: tuple[int, int, int, int, int, int],
    token_block: int,
    feature_wave_blocks: int,
    input_dtype: str,
    output_dtype: str | None = None,
    pair_layout_id: str,
    device: str,
    projected_cache_after_manifest: tuple[PairMetadataCacheEntry, ...],
    requested_gradient_mask: tuple[bool, bool, bool, bool, bool, bool],
    precision: str = "fp32_ieee",
    result_contract: str = "full_aux",
    key_feature_impl: str = "generic_materialized",
    key_fold_impl: str = "generic_materialized",
    query_feature_impl: str = "generic_materialized",
    query_fold_impl: str = "generic_materialized",
    query_fold_input: str = "staged_fp32",
    query_gradient_flow: str = "materialized",
    query_producer_fold_strategy: str | None = None,
    query_producer_partition_count: int | None = None,
    query_consumer_stages: Sequence[str] | None = None,
    query_consumer_fusion: str | None = None,
    backward_schedule: str = "split",
    gradient_staging: str = "per_wave",
    backward_normalize_impl: str = "torch",
    save_local_score: bool = False,
    key_retention: str = "none",
    physical_feature_dimension: int | None = None,
) -> tuple[HDLogicalBuffer, ...]:
    """Compile the sole canonical physical allocation table for a plan."""
    if precision not in ("fp32_ieee", "bf16_tensorcore"):
        raise ValueError("unsupported HD Block-GEMM precision")
    if output_dtype is None:
        output_dtype = input_dtype
    master_storage = _FEATURE_STORAGE
    tc_storage = "bfloat16" if precision == "bf16_tensorcore" else master_storage
    tc_enabled = precision == "bf16_tensorcore"
    (
        batch_size,
        query_heads,
        key_value_heads,
        sequence_length,
        head_dimension,
        value_dimension,
    ) = geometry
    canonical_query_config = _validate_query_operator_identity(
        query_feature_impl=query_feature_impl,
        query_fold_impl=query_fold_impl,
        query_gradient_flow=query_gradient_flow,
        query_producer_fold_strategy=query_producer_fold_strategy,
        query_consumer_stages=query_consumer_stages,
        query_consumer_fusion=query_consumer_fusion,
        precision=precision,
        head_dimension=head_dimension,
        result_contract=result_contract,
    )
    query_feature_impl = str(canonical_query_config["query_feature_impl"])
    query_fold_impl = str(canonical_query_config["query_fold_impl"])
    query_gradient_flow = str(canonical_query_config["query_gradient_flow"])
    query_consumer_stages = tuple(canonical_query_config["query_consumer_stages"])
    key_feature_impl = _validate_key_feature_impl(
        key_feature_impl=key_feature_impl,
        precision=precision,
        head_dimension=head_dimension,
        result_contract=result_contract,
    )
    key_fold_impl = _validate_key_fold_impl(
        key_fold_impl=key_fold_impl,
        precision=precision,
        head_dimension=head_dimension,
        result_contract=result_contract,
    )
    backward_normalize_impl = _validate_backward_normalize_impl(
        backward_normalize_impl,
        precision=precision,
    )
    if query_producer_partition_count is not None:
        raise ValueError("query_producer_partition_count: OP-3 exploration is retired")
    pair_count = head_dimension * (head_dimension + 1) // 2
    number_blocks = ceil(sequence_length / token_block)
    feature_dimension = 1 + head_dimension + pair_count
    if physical_feature_dimension is None:
        physical_feature_dimension = feature_dimension
    if physical_feature_dimension < feature_dimension:
        raise ValueError("physical feature dimension cannot be smaller than logical")
    augmented_value_dimension = value_dimension + 1
    wave_tokens = feature_wave_blocks * token_block
    backward_requested = any(requested_gradient_mask)
    need_q, need_k, need_v, need_a, need_b, need_c = requested_gradient_mask
    query_side = need_q or need_a or need_b or need_c
    right_scan = need_k or need_v
    backward_schedule = _validate_backward_schedule(
        backward_schedule,
        requested_gradient_mask=requested_gradient_mask,
        query_gradient_flow=query_gradient_flow,
        query_consumer_stages=query_consumer_stages,
    )
    gradient_staging = _validate_gradient_staging(
        gradient_staging,
        precision=precision,
    )
    save_local_score = _validate_save_local_score(
        save_local_score,
        requested_gradient_mask=requested_gradient_mask,
        precision=precision,
    )
    key_retention = _validate_key_retention(
        key_retention,
        precision=precision,
        requested_gradient_mask=requested_gradient_mask,
    )
    shared_wave = backward_schedule == "shared_wave"
    query_compute_stage = (
        "backward_shared_wave" if shared_wave else "backward_query_wave"
    )
    kv_compute_stage = "backward_shared_wave" if shared_wave else "backward_kv_wave"
    backward_recompute_end_stage = (
        "backward_shared_wave"
        if shared_wave
        else ("backward_kv_wave" if right_scan else "backward_query_wave")
    )
    normalization_end_stage = (
        "backward_shared_wave"
        if shared_wave
        else ("backward_kv_wave" if right_scan else "backward_query_wave")
    )
    last_forward_stage = "forward_outputs_saved"
    last_stage = (
        "backward_shared_wave"
        if shared_wave
        else ("backward_kv_wave" if backward_requested else last_forward_stage)
    )

    q_shape = (batch_size, query_heads, sequence_length, head_dimension)
    k_shape = (batch_size, key_value_heads, sequence_length, head_dimension)
    v_shape = (batch_size, key_value_heads, sequence_length, value_dimension)
    output_shape = (batch_size, query_heads, sequence_length, value_dimension)
    state_shape = (
        batch_size,
        key_value_heads,
        number_blocks,
        physical_feature_dimension,
        augmented_value_dimension,
    )
    phi_q_wave_shape = (
        batch_size,
        query_heads,
        wave_tokens,
        physical_feature_dimension,
    )
    phi_k_wave_shape = (
        batch_size,
        key_value_heads,
        wave_tokens,
        physical_feature_dimension,
    )
    phi_k_cache_shape = (
        ceil(number_blocks / feature_wave_blocks),
        batch_size,
        key_value_heads,
        feature_wave_blocks,
        token_block,
        physical_feature_dimension,
    )
    phi_q_input_shape = (
        batch_size,
        query_heads,
        wave_tokens,
        head_dimension,
    )
    phi_k_input_shape = (
        batch_size,
        key_value_heads,
        wave_tokens,
        head_dimension,
    )
    phi_q_pair_shape = (batch_size, query_heads, wave_tokens, pair_count)
    phi_k_pair_shape = (batch_size, key_value_heads, wave_tokens, pair_count)
    need_generic_query_features = (
        right_scan and query_feature_impl == "generic_materialized"
    )
    need_generic_key_features = (
        query_side or need_v
    ) and key_feature_impl == "generic_materialized"
    backward_feature_pair_heads = max(
        query_heads if need_generic_query_features else 0,
        key_value_heads if need_generic_key_features else 0,
    )
    backward_feature_pair_shape = (
        (
            batch_size,
            backward_feature_pair_heads,
            wave_tokens,
            pair_count,
        )
        if backward_feature_pair_heads
        else None
    )
    need_generic_forward_query_features = query_feature_impl == "generic_materialized"
    need_generic_forward_key_features = key_feature_impl == "generic_materialized"
    forward_pair_heads = max(
        query_heads if need_generic_forward_query_features else 0,
        key_value_heads if need_generic_forward_key_features else 0,
    )
    forward_pair_shape = (
        (batch_size, forward_pair_heads, wave_tokens, pair_count)
        if forward_pair_heads
        else None
    )
    query_fold_output_shape = (
        batch_size,
        query_heads,
        wave_tokens,
        head_dimension,
    )
    key_fold_output_shape = (
        batch_size,
        key_value_heads,
        wave_tokens,
        head_dimension,
    )
    u_wave_shape = (
        batch_size,
        key_value_heads,
        wave_tokens,
        augmented_value_dimension,
    )
    local_score_shape = (
        batch_size,
        query_heads,
        feature_wave_blocks,
        token_block,
        token_block,
    )
    local_score_cache_shape = (
        ceil(number_blocks / feature_wave_blocks),
        batch_size,
        query_heads,
        feature_wave_blocks,
        token_block,
        token_block,
    )
    y_wave_shape = (
        batch_size,
        query_heads,
        wave_tokens,
        augmented_value_dimension,
    )
    block_totals_wave_shape = (
        batch_size,
        key_value_heads,
        feature_wave_blocks,
        physical_feature_dimension,
        augmented_value_dimension,
    )
    carry_query_wave_shape = (
        batch_size,
        query_heads,
        feature_wave_blocks,
        physical_feature_dimension,
        augmented_value_dimension,
    )
    phi_k_query_wave_shape = (
        batch_size,
        query_heads,
        wave_tokens,
        physical_feature_dimension,
    )
    u_query_wave_shape = (
        batch_size,
        query_heads,
        wave_tokens,
        augmented_value_dimension,
    )

    buffers: list[HDLogicalBuffer] = []

    def add(
        start_stage: str,
        end_stage: str,
        *declarations: tuple[str, tuple[int, ...], str],
    ) -> None:
        """Declare a group with one inclusive lifetime, in canonical order."""
        buffers.extend(
            HDLogicalBuffer(name, shape, dtype, start_stage, end_stage)
            for name, shape, dtype in declarations
        )

    # Metadata and forward-owned storage.
    current_cache_key = (pair_layout_id, device)
    for entry in projected_cache_after_manifest:
        if entry.key == current_cache_key:
            names = tuple(
                f"pair_{component}_cache"
                for component in ("rows", "columns", "multiplicity")
            )
        else:
            entry_id = _sha256({"layout_id": entry.layout_id, "device": entry.device})
            names = tuple(
                f"pair_cache_entry_{entry_id}_{component}"
                for component in ("rows", "columns", "multiplicity")
            )
        add(
            "forward_totals",
            last_stage,
            *((name, (entry.pair_count,), "int64") for name in names),
        )
    add(
        "forward_totals",
        last_stage,
        *(
            (f"pair_{component}_materialized", (pair_count,), "int64")
            for component in ("rows", "columns", "multiplicity")
        ),
    )

    save_a = need_k or need_v
    save_b_c = need_q or need_k or need_v
    save_scale = need_q or need_k or need_v or need_b or need_c
    for name, shape, saved in (
        ("logical_a", (query_heads,), save_a),
        ("logical_b", (query_heads, head_dimension), save_b_c),
        ("logical_c", (query_heads, pair_count), save_b_c),
        ("logical_scale", (1,), save_scale),
    ):
        add(
            "forward_totals",
            last_stage if saved else last_forward_stage,
            (name, shape, master_storage),
        )

    if backward_requested:
        q_needed_in_backward = need_q or need_k or need_v or need_b or need_c
        add(
            "forward_totals",
            last_stage if q_needed_in_backward else last_forward_stage,
            ("saved_q" if q_needed_in_backward else "q_input", q_shape, input_dtype),
        )
        add("forward_totals", last_stage, ("saved_k", k_shape, input_dtype))
        v_saved = need_q or need_k or need_a or need_b or need_c
        add(
            "forward_totals",
            last_stage if v_saved else last_forward_stage,
            ("saved_v" if v_saved else "v_input", v_shape, input_dtype),
        )
    else:
        add(
            "forward_totals",
            last_forward_stage,
            ("q_input", q_shape, input_dtype),
            ("k_input", k_shape, input_dtype),
            ("v_input", v_shape, input_dtype),
        )

    output_end = last_stage if backward_requested else last_forward_stage
    add(
        "forward_feature_wave",
        output_end,
        ("output", output_shape, output_dtype),
        ("numerator", output_shape, master_storage),
        ("denominator", (batch_size, query_heads, sequence_length, 1), master_storage),
    )
    add(
        "forward_totals",
        last_stage if query_side else "forward_feature_wave",
        ("block_totals_carry", state_shape, master_storage),
    )
    if key_retention != "none":
        phi_k_end = (
            kv_compute_stage
            if key_retention == "backward" and need_v
            else (
                query_compute_stage
                if key_retention == "backward"
                else "forward_feature_wave"
            )
        )
        add(
            "forward_totals",
            phi_k_end,
            (
                (
                    "saved_phi_k_tc"
                    if key_retention == "backward"
                    else "forward_phi_k_cache"
                ),
                phi_k_cache_shape,
                "bfloat16",
            ),
        )
    else:
        add(
            "forward_totals",
            "forward_feature_wave",
            ("forward_phi_k_wave", phi_k_wave_shape, tc_storage),
        )
    add(
        "forward_totals",
        "forward_feature_wave",
        ("forward_phi_q_wave", phi_q_wave_shape, tc_storage),
    )
    if need_generic_forward_query_features:
        add(
            "forward_totals",
            "forward_feature_wave",
            ("forward_phi_q_input_work", phi_q_input_shape, master_storage),
        )
    if forward_pair_shape is not None:
        add(
            "forward_totals",
            "forward_feature_wave",
            ("forward_pair_work", forward_pair_shape, master_storage),
        )
    add(
        "forward_totals",
        "forward_feature_wave",
        ("forward_u_wave", u_wave_shape, tc_storage),
    )
    add(
        "forward_totals",
        "forward_totals",
        ("forward_block_totals_wave", block_totals_wave_shape, master_storage),
    )
    add(
        "forward_carry_scan",
        "forward_carry_scan",
        ("forward_scan_inclusive", state_shape, master_storage),
    )
    add(
        "forward_feature_wave",
        "forward_feature_wave",
        ("forward_local_score", local_score_shape, master_storage),
        ("forward_y_wave", y_wave_shape, master_storage),
    )
    if save_local_score:
        add(
            "forward_feature_wave",
            kv_compute_stage,
            ("saved_local_score_tc", local_score_cache_shape, "bfloat16"),
        )
    if tc_enabled:
        if need_generic_forward_key_features:
            add(
                "forward_totals",
                "forward_feature_wave",
                ("forward_phi_k_input_work", phi_k_input_shape, master_storage),
            )
        if forward_pair_shape is not None:
            add(
                "forward_totals",
                "forward_feature_wave",
                ("forward_pair_second_work", forward_pair_shape, master_storage),
            )
        if not save_local_score:
            add(
                "forward_feature_wave",
                "forward_feature_wave",
                ("forward_local_score_tc", local_score_shape, tc_storage),
            )
        add(
            "forward_feature_wave",
            "forward_feature_wave",
            ("forward_local_y", y_wave_shape, master_storage),
        )
    add(
        "forward_feature_wave",
        "forward_feature_wave",
        ("forward_carry_query_wave", carry_query_wave_shape, tc_storage),
    )
    if query_heads > key_value_heads:
        add(
            "forward_feature_wave",
            "forward_feature_wave",
            ("forward_phi_k_query_wave", phi_k_query_wave_shape, tc_storage),
            ("forward_u_query_wave", u_query_wave_shape, tc_storage),
        )

    if not backward_requested:
        return tuple(buffers)

    # Normalization and feature recomputation storage.
    normalization_denominator_shape = (batch_size, query_heads, sequence_length, 1)
    add(
        "backward_normalization",
        normalization_end_stage,
        (
            "backward_normalization_g",
            (batch_size, query_heads, sequence_length, augmented_value_dimension),
            master_storage,
        ),
    )
    if backward_normalize_impl == "torch":
        add(
            "backward_normalization",
            "backward_normalization",
            (
                "backward_normalization_denominator_work",
                normalization_denominator_shape,
                master_storage,
            ),
            ("backward_normalization_active", normalization_denominator_shape, "bool"),
        )
    if backward_feature_pair_shape is not None:
        add(
            "backward_query_wave",
            backward_recompute_end_stage,
            ("backward_feature_pair_work", backward_feature_pair_shape, master_storage),
        )
    if gradient_staging == "full_bf16":
        add(
            "backward_normalization",
            normalization_end_stage,
            (
                "backward_g_full_bf16",
                (
                    ceil(number_blocks / feature_wave_blocks),
                    batch_size,
                    query_heads,
                    feature_wave_blocks,
                    token_block,
                    augmented_value_dimension,
                ),
                "bfloat16",
            ),
        )
    else:
        add(
            "backward_query_wave",
            backward_recompute_end_stage,
            ("backward_g_wave", y_wave_shape, tc_storage),
        )
    if right_scan:
        add(
            "backward_query_wave",
            kv_compute_stage,
            ("backward_phi_q_wave", phi_q_wave_shape, tc_storage),
        )
    if query_side or need_v:
        phi_k_start = query_compute_stage if query_side else kv_compute_stage
        phi_k_end = kv_compute_stage if need_v else query_compute_stage
        if key_retention != "backward":
            add(
                phi_k_start,
                phi_k_end,
                ("backward_phi_k_wave", phi_k_wave_shape, tc_storage),
            )
        if query_heads > key_value_heads:
            add(
                phi_k_start,
                phi_k_end,
                ("backward_phi_k_query_wave", phi_k_query_wave_shape, tc_storage),
            )
    if query_side or need_k:
        u_start = query_compute_stage if query_side else kv_compute_stage
        u_end = kv_compute_stage if need_k else query_compute_stage
        add(
            u_start,
            u_end,
            ("backward_u_wave", u_wave_shape, tc_storage),
            ("backward_ds_local", local_score_shape, master_storage),
        )
        if query_heads > key_value_heads:
            add(
                u_start,
                u_end,
                ("backward_u_query_wave", u_query_wave_shape, tc_storage),
            )
    if query_side:
        add(
            query_compute_stage,
            query_compute_stage,
            ("backward_carry_query_wave", carry_query_wave_shape, tc_storage),
        )
    for requested, name, shape, stage in (
        (
            need_v and not save_local_score,
            "backward_local_score",
            local_score_shape,
            kv_compute_stage,
        ),
        (query_side, "backward_dphi_q", phi_q_wave_shape, query_compute_stage),
        (need_k, "backward_dphi_k", phi_k_wave_shape, kv_compute_stage),
        (need_v, "backward_du", u_wave_shape, kv_compute_stage),
    ):
        if requested:
            add(stage, stage, (name, shape, master_storage))
    if tc_enabled:
        if backward_feature_pair_shape is not None:
            add(
                "backward_query_wave",
                backward_recompute_end_stage,
                (
                    "backward_feature_pair_second_work",
                    backward_feature_pair_shape,
                    master_storage,
                ),
            )
        if need_generic_key_features and key_retention != "backward":
            phi_k_start = query_compute_stage if query_side else kv_compute_stage
            phi_k_end = kv_compute_stage if need_v else query_compute_stage
            add(
                phi_k_start,
                phi_k_end,
                ("backward_feature_k_input_work", phi_k_input_shape, master_storage),
            )
        if query_side or need_k:
            ds_start = query_compute_stage if query_side else kv_compute_stage
            ds_end = kv_compute_stage if need_k else query_compute_stage
            add(
                ds_start,
                ds_end,
                ("backward_ds_local_tc", local_score_shape, tc_storage),
            )
        for requested, name, shape, dtype, stage in (
            (
                query_side,
                "backward_dphi_q_local",
                phi_q_wave_shape,
                master_storage,
                query_compute_stage,
            ),
            (
                need_k,
                "backward_dphi_k_cross",
                phi_k_wave_shape,
                master_storage,
                kv_compute_stage,
            ),
            (
                need_v and not save_local_score,
                "backward_local_score_tc",
                local_score_shape,
                tc_storage,
                kv_compute_stage,
            ),
            (
                need_v,
                "backward_du_cross",
                u_wave_shape,
                master_storage,
                kv_compute_stage,
            ),
            (
                right_scan,
                "backward_dt_wave_tc",
                block_totals_wave_shape,
                tc_storage,
                kv_compute_stage,
            ),
        ):
            if requested:
                add(stage, stage, (name, shape, dtype))
    if query_heads > key_value_heads:
        if need_k:
            add(
                kv_compute_stage,
                kv_compute_stage,
                ("backward_dphi_k_query_wave", phi_k_query_wave_shape, master_storage),
            )
        if need_v:
            add(
                kv_compute_stage,
                kv_compute_stage,
                ("backward_du_query_wave", u_query_wave_shape, master_storage),
            )

    # Fold workspaces and saved right-scan state.
    needs_materialized_query_input = (
        query_gradient_flow == "materialized"
        and query_fold_input == "staged_fp32"
        and (need_q or need_b or need_c)
    )
    needs_generic_query_feature_input = (
        query_feature_impl == "generic_materialized" and right_scan
    )
    if needs_materialized_query_input or needs_generic_query_feature_input:
        query_input_start = (
            "backward_query_wave"
            if needs_generic_query_feature_input
            else query_compute_stage
        )
        add(
            query_input_start,
            backward_recompute_end_stage,
            ("backward_query_input_work", phi_q_input_shape, master_storage),
        )
    if query_fold_impl == "generic_materialized":
        if need_q or need_c:
            add(
                query_compute_stage,
                last_stage,
                ("backward_query_fold_source", phi_q_pair_shape, master_storage),
            )
        if need_c:
            add(
                query_compute_stage,
                last_stage,
                ("backward_query_fold_second", phi_q_pair_shape, master_storage),
            )
        if need_b:
            add(
                query_compute_stage,
                last_stage,
                (
                    "backward_query_fold_linear_source",
                    phi_q_input_shape,
                    master_storage,
                ),
            )
    if need_q:
        add(
            query_compute_stage,
            last_stage,
            ("backward_query_fold_output", query_fold_output_shape, master_storage),
        )
    if need_k:
        if key_fold_impl == "generic_materialized":
            add(
                kv_compute_stage,
                last_stage,
                ("backward_key_input_work", phi_k_input_shape, master_storage),
                ("backward_key_fold_source", phi_k_pair_shape, master_storage),
            )
        add(
            kv_compute_stage,
            last_stage,
            ("backward_key_fold_output", key_fold_output_shape, master_storage),
        )
    if right_scan:
        add(
            "backward_query_wave",
            "backward_query_wave",
            ("backward_dcarry_query_wave", carry_query_wave_shape, master_storage),
        )
        add(
            "backward_query_wave",
            "backward_right_scan",
            ("backward_dcarry", state_shape, master_storage),
        )
        add(
            "backward_right_scan",
            kv_compute_stage,
            ("backward_right_scan_dt", state_shape, master_storage),
        )
        add(
            "backward_right_scan",
            "backward_right_scan",
            (
                "backward_right_scan_total",
                (
                    batch_size,
                    key_value_heads,
                    1,
                    physical_feature_dimension,
                    augmented_value_dimension,
                ),
                master_storage,
            ),
        )

    # Final gradients and coefficient partial reductions.
    gradient_storage = master_storage if tc_enabled else input_dtype
    for requested, name, shape, stage in (
        (need_q, "grad_q", q_shape, query_compute_stage),
        (need_k, "grad_k", k_shape, kv_compute_stage),
        (need_v, "grad_v", v_shape, kv_compute_stage),
    ):
        if requested:
            add(stage, last_stage, (name, shape, gradient_storage))
    for requested, suffix, partial_shape, gradient_shape in (
        (need_a, "a", (batch_size, number_blocks, query_heads), (query_heads,)),
        (
            need_b,
            "b",
            (batch_size, number_blocks, query_heads, head_dimension),
            (query_heads, head_dimension),
        ),
        (
            need_c,
            "c",
            (batch_size, number_blocks, query_heads, pair_count),
            (query_heads, pair_count),
        ),
    ):
        if requested:
            add(
                query_compute_stage,
                query_compute_stage,
                (
                    f"backward_coefficient_{suffix}_block_partials",
                    partial_shape,
                    master_storage,
                ),
            )
            add(
                query_compute_stage,
                last_stage,
                (f"grad_{suffix}", gradient_shape, master_storage),
            )

    return tuple(buffers)
