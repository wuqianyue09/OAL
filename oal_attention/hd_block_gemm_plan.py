"""Immutable HD Block-GEMM plan values, validation and stable serialization.

Tensor-input inspection and backend resolution remain in hd_block_gemm. The
cache depends only on the nominal contracts, keeping value imports acyclic.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

from .hd_cublas_compat import HdContractionBackendIdentity
from .hd_block_gemm_buffers import (
    HDLogicalBuffer,
    _EXECUTION_STAGES,
    _FEATURE_STORAGE,
    _byte_summaries,
    _expected_logical_buffers,
)
from .hd_block_gemm_cache import (
    PairMetadataCacheEntry,
    _canonical_runtime_device,
    _project_cache_manifest,
    canonical_pair_layout,
)
from .hd_block_gemm_contracts import (
    _HDParallelBlockPlanContract,
    _contiguous_strides,
    _require_plain_int,
    _sha256,
    _stable_json,
    _validate_backward_schedule,
    _validate_backward_normalize_impl,
    _validate_kv_cross_impl,
    _validate_query_feature_token_tile,
    _validate_query_fold_token_tile,
    _validate_query_fold_input,
    _validate_gradient_staging,
    _validate_forward_normalize_impl,
    _validate_feature_padding,
    _validate_key_feature_impl,
    _validate_key_fold_impl,
    _validate_query_operator_identity,
    _validate_key_retention,
    _validate_save_local_score,
    legacy_query_consumer_fusion_for_stages,
)

_PHYSICAL_PATH = "causal_parallel_block_gemm"
_INPUT_LAYOUT = "strided_contiguous"
_SUPPORTED_PRECISIONS = frozenset(("fp32_ieee", "bf16_tensorcore"))
_INPUT_DTYPE_NAMES = frozenset(("float16", "bfloat16", "float32", "float64"))
_FULL_AUX_RESULT_CONTRACT = "full_aux"
_NORMALIZED_OUTPUT_ONLY_RESULT_CONTRACT = "normalized_output_only"
_SUPPORTED_RESULT_CONTRACTS = frozenset(
    (_FULL_AUX_RESULT_CONTRACT, _NORMALIZED_OUTPUT_ONLY_RESULT_CONTRACT)
)


def _feature_storage_for_precision(precision: str) -> str:
    if precision == "fp32_ieee":
        return _FEATURE_STORAGE
    if precision == "bf16_tensorcore":
        return "bfloat16"
    raise ValueError("unsupported HD Block-GEMM precision")


def _validate_cache_manifest(
    manifest: object,
    *,
    name: str,
) -> tuple[PairMetadataCacheEntry, ...]:
    if not isinstance(manifest, tuple) or any(
        not isinstance(entry, PairMetadataCacheEntry) for entry in manifest
    ):
        raise TypeError(f"{name} must be a tuple of PairMetadataCacheEntry values")
    keys = tuple(entry.key for entry in manifest)
    if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
        raise ValueError(f"{name} must be unique and sorted by cache key")
    return manifest


@dataclass(frozen=True)
class HDParallelBlockPlan(_HDParallelBlockPlanContract):
    """Frozen, deterministic physical plan for one Q/K/V geometry."""

    physical_path: str
    token_block: int
    feature_wave_blocks: int
    feature_storage: str
    result_contract: str
    key_feature_impl: str
    key_fold_impl: str
    query_feature_impl: str
    query_feature_token_tile: int
    query_fold_impl: str
    query_fold_token_tile: int
    query_fold_input: str
    query_gradient_flow: str
    query_producer_fold_strategy: str
    query_producer_partition_count: int | None
    query_consumer_stages: tuple[str, ...]
    backward_schedule: str
    gradient_staging: str
    forward_normalize_impl: str
    backward_normalize_impl: str
    kv_cross_impl: str
    requested_precision: str
    precision: str
    precision_fallback_reason: str | None
    contraction_backend_identity: HdContractionBackendIdentity
    requested_gradient_mask: tuple[bool, bool, bool, bool, bool, bool]
    geometry: tuple[int, int, int, int, int, int]
    gqa_ratio: int
    pair_layout_id: str
    pair_count: int
    feature_dimension: int
    feature_padding: str
    physical_feature_dimension: int
    augmented_value_dimension: int
    physical_augmented_value_dimension: int
    number_blocks: int
    input_dtype: str
    input_layout: str
    input_strides: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    device_type: str
    device: str
    cache_before_manifest: tuple[PairMetadataCacheEntry, ...]
    projected_cache_after_manifest: tuple[PairMetadataCacheEntry, ...]
    memory_budget_bytes: int | None
    model_residency_bytes: int
    execution_stages: tuple[str, ...]
    logical_buffers: tuple[HDLogicalBuffer, ...]
    forward_block_totals_bytes: int
    saved_carry_bytes: int
    output_bytes: int
    largest_feature_wave_bytes: int
    local_score_bytes: int
    forward_ephemeral_peak_bytes: int
    saved_tensor_bytes: int
    saved_coefficient_bytes: int
    saved_activation_bytes: int
    cache_before_residency_bytes: int
    core_metadata_residency_bytes: int
    pair_metadata_materialization_bytes: int
    backward_right_scan_bytes: int
    backward_scratch_peak_bytes: int
    invocation_peak_bytes: int
    projected_live_bytes: int
    save_local_score: bool = False
    requested_key_retention: str = "none"
    key_retention: str = "none"
    plan_id: str = ""

    def __post_init__(self) -> None:
        _require_plain_int(self.token_block, name="token_block", minimum=1)
        _require_plain_int(
            self.feature_wave_blocks,
            name="feature_wave_blocks",
            minimum=1,
        )
        if self.physical_path != _PHYSICAL_PATH:
            raise ValueError("physical path must be causal_parallel_block_gemm")
        if self.result_contract not in _SUPPORTED_RESULT_CONTRACTS:
            raise ValueError("unsupported HD Block-GEMM result contract")
        if self.result_contract != _FULL_AUX_RESULT_CONTRACT:
            raise NotImplementedError(
                "normalized_output_only is an unimplemented private specialization"
            )
        if self.requested_precision not in _SUPPORTED_PRECISIONS:
            raise ValueError("unsupported requested HD Block-GEMM precision")
        if self.precision not in _SUPPORTED_PRECISIONS:
            raise ValueError("unsupported HD Block-GEMM precision")
        if not isinstance(
            self.contraction_backend_identity,
            HdContractionBackendIdentity,
        ):
            raise TypeError("plan contraction backend identity is invalid")
        if self.requested_precision == "fp32_ieee":
            if (
                self.precision != "fp32_ieee"
                or self.precision_fallback_reason is not None
            ):
                raise ValueError("FP32 requests cannot carry a precision fallback")
        elif self.precision == "bf16_tensorcore":
            if self.precision_fallback_reason is not None:
                raise ValueError("verified BF16 plans cannot carry a fallback reason")
        elif (
            self.precision != "fp32_ieee"
            or self.precision_fallback_reason != "no_verified_bf16_contraction_backend"
        ):
            raise ValueError("ordinary BF16 fallback identity is invalid")
        backend_kind = self.contraction_backend_identity.backend_kind
        if self.precision == "fp32_ieee" and backend_kind != "fp32_ieee":
            raise ValueError("FP32 plan must bind the builtin FP32 backend")
        if self.precision == "bf16_tensorcore" and backend_kind not in (
            "torch_bmm_out_dtype",
            "cublas_strided_batched_ex",
        ):
            raise ValueError("BF16 plan must bind a verified BF16 backend")
        if self.feature_storage != _feature_storage_for_precision(self.precision):
            raise ValueError("HD Block-GEMM feature storage does not match precision")
        if self.precision == "bf16_tensorcore" and (
            self.device_type != "cuda" or self.input_dtype != "bfloat16"
        ):
            raise ValueError("bf16_tensorcore requires CUDA bfloat16 inputs")
        if self.execution_stages != _EXECUTION_STAGES:
            raise ValueError("plan execution stages do not match the physical contract")
        if (
            not isinstance(self.geometry, tuple)
            or len(self.geometry) != 6
            or any(
                not isinstance(dimension, int)
                or isinstance(dimension, bool)
                or dimension <= 0
                for dimension in self.geometry
            )
        ):
            raise ValueError("plan geometry must contain six positive dimensions")
        canonical_query_config = _validate_query_operator_identity(
            query_feature_impl=self.query_feature_impl,
            query_fold_impl=self.query_fold_impl,
            query_gradient_flow=self.query_gradient_flow,
            query_producer_fold_strategy=self.query_producer_fold_strategy,
            query_consumer_stages=self.query_consumer_stages,
            precision=self.precision,
            head_dimension=self.head_dimension,
            result_contract=self.result_contract,
        )
        canonical_key_feature_impl = _validate_key_feature_impl(
            key_feature_impl=self.key_feature_impl,
            precision=self.precision,
            head_dimension=self.head_dimension,
            result_contract=self.result_contract,
        )
        canonical_key_fold_impl = _validate_key_fold_impl(
            key_fold_impl=self.key_fold_impl,
            precision=self.precision,
            head_dimension=self.head_dimension,
            result_contract=self.result_contract,
        )
        if self.key_feature_impl != canonical_key_feature_impl:
            raise ValueError("plan key feature selector is not canonical")
        if self.key_fold_impl != canonical_key_fold_impl:
            raise ValueError("plan key fold selector is not canonical")
        if (
            self.query_feature_impl != canonical_query_config["query_feature_impl"]
            or self.query_fold_impl != canonical_query_config["query_fold_impl"]
            or self.query_gradient_flow != canonical_query_config["query_gradient_flow"]
            or self.query_producer_fold_strategy
            != canonical_query_config["query_producer_fold_strategy"]
            or self.query_consumer_stages
            != tuple(canonical_query_config["query_consumer_stages"])
        ):
            raise ValueError("plan query operator selectors are not canonical")
        canonical_query_feature_token_tile = _validate_query_feature_token_tile(
            self.query_feature_token_tile,
            query_feature_impl=self.query_feature_impl,
        )
        if self.query_feature_token_tile != canonical_query_feature_token_tile:
            raise ValueError("plan query feature token tile is not canonical")
        canonical_query_fold_token_tile = _validate_query_fold_token_tile(
            self.query_fold_token_tile,
            query_fold_impl=self.query_fold_impl,
        )
        if self.query_fold_token_tile != canonical_query_fold_token_tile:
            raise ValueError("plan query fold token tile is not canonical")
        canonical_query_fold_input = _validate_query_fold_input(
            self.query_fold_input,
            query_fold_impl=self.query_fold_impl,
            query_gradient_flow=self.query_gradient_flow,
        )
        if self.query_fold_input != canonical_query_fold_input:
            raise ValueError("plan query fold input is not canonical")
        canonical_backward_schedule = _validate_backward_schedule(
            self.backward_schedule,
            requested_gradient_mask=self.requested_gradient_mask,
            query_gradient_flow=self.query_gradient_flow,
            query_consumer_stages=self.query_consumer_stages,
        )
        if self.backward_schedule != canonical_backward_schedule:
            raise ValueError("plan backward schedule is not canonical")
        canonical_gradient_staging = _validate_gradient_staging(
            self.gradient_staging,
            precision=self.precision,
        )
        if self.gradient_staging != canonical_gradient_staging:
            raise ValueError("plan gradient staging selector is not canonical")
        canonical_forward_normalize_impl = _validate_forward_normalize_impl(
            self.forward_normalize_impl,
            precision=self.precision,
        )
        if self.forward_normalize_impl != canonical_forward_normalize_impl:
            raise ValueError("plan forward normalization selector is not canonical")
        canonical_backward_normalize_impl = _validate_backward_normalize_impl(
            self.backward_normalize_impl,
            precision=self.precision,
        )
        if self.backward_normalize_impl != canonical_backward_normalize_impl:
            raise ValueError("plan backward normalization selector is not canonical")
        canonical_kv_cross_impl = _validate_kv_cross_impl(
            self.kv_cross_impl,
            precision=self.precision,
        )
        if self.kv_cross_impl != canonical_kv_cross_impl:
            raise ValueError("plan KV-cross selector is not canonical")
        canonical_save_local_score = _validate_save_local_score(
            self.save_local_score,
            requested_gradient_mask=self.requested_gradient_mask,
            precision=self.precision,
        )
        if self.save_local_score != canonical_save_local_score:
            raise ValueError("plan saved local score selector is not canonical")
        canonical_requested_key_retention = _validate_key_retention(
            self.requested_key_retention,
            precision=self.precision,
        )
        if self.requested_key_retention != canonical_requested_key_retention:
            raise ValueError("plan requested key retention selector is not canonical")
        canonical_key_retention = _validate_key_retention(
            canonical_requested_key_retention,
            precision=self.precision,
            requested_gradient_mask=self.requested_gradient_mask,
        )
        if self.key_retention != canonical_key_retention:
            raise ValueError("plan key retention selector is not canonical")
        if self.query_producer_partition_count is not None:
            raise ValueError(
                "query_producer_partition_count: OP-3 exploration is retired"
            )
        if (
            not isinstance(self.requested_gradient_mask, tuple)
            or len(self.requested_gradient_mask) != 6
            or any(
                type(requested) is not bool
                for requested in self.requested_gradient_mask
            )
        ):
            raise ValueError("plan gradient mask must be a tuple of six bools")
        if self.query_heads % self.key_value_heads:
            raise ValueError("plan geometry requires integral GQA ownership")
        if self.gqa_ratio != self.query_heads // self.key_value_heads:
            raise ValueError("plan GQA ratio does not match its geometry")
        layout = canonical_pair_layout(self.head_dimension)
        if (
            self.pair_layout_id != layout.layout_id
            or self.pair_count != layout.pair_count
        ):
            raise ValueError("plan pair metadata does not match its head dimension")
        if self.feature_dimension != 1 + self.head_dimension + self.pair_count:
            raise ValueError("plan feature dimension does not match its pair basis")
        canonical_feature_padding = _validate_feature_padding(
            self.feature_padding,
            head_dimension=self.head_dimension,
        )
        expected_physical_feature_dimension = (
            2160 if canonical_feature_padding == "f2160" else self.feature_dimension
        )
        if self.physical_feature_dimension != expected_physical_feature_dimension:
            raise ValueError(
                "plan physical feature dimension does not match its selector"
            )
        if self.augmented_value_dimension != self.value_dimension + 1:
            raise ValueError("plan augmented value dimension must equal DV + 1")
        if self.physical_augmented_value_dimension != self.augmented_value_dimension:
            raise ValueError(
                "plan physical augmented value dimension must remain logical"
            )
        if self.number_blocks != ceil(self.sequence_length / self.token_block):
            raise ValueError("plan number of blocks does not match N and token_block")
        if self.feature_wave_blocks > self.number_blocks:
            raise ValueError("plan feature wave cannot exceed its number of blocks")
        if self.input_dtype not in _INPUT_DTYPE_NAMES:
            raise ValueError("plan input dtype is unsupported")
        if self.input_layout != _INPUT_LAYOUT:
            raise ValueError("plan input layout must be strided_contiguous")
        expected_input_strides = tuple(
            _contiguous_strides(shape)
            for shape in (
                (
                    self.batch_size,
                    self.query_heads,
                    self.sequence_length,
                    self.head_dimension,
                ),
                (
                    self.batch_size,
                    self.key_value_heads,
                    self.sequence_length,
                    self.head_dimension,
                ),
                (
                    self.batch_size,
                    self.key_value_heads,
                    self.sequence_length,
                    self.value_dimension,
                ),
            )
        )
        if self.input_strides != expected_input_strides:
            raise ValueError("plan input strides must be exact row-major contiguous")
        plan_device = _canonical_runtime_device(self.device, name="plan device")
        if plan_device.type != self.device_type:
            raise ValueError("plan device identity does not match its device type")
        cache_before_manifest = _validate_cache_manifest(
            self.cache_before_manifest,
            name="cache_before_manifest",
        )
        projected_cache_after_manifest = _validate_cache_manifest(
            self.projected_cache_after_manifest,
            name="projected_cache_after_manifest",
        )
        current_cache_entry = PairMetadataCacheEntry(
            layout_id=self.pair_layout_id,
            device=self.device,
            pair_count=self.pair_count,
        )
        expected_after = _project_cache_manifest(
            cache_before_manifest,
            current_cache_entry,
        )
        if projected_cache_after_manifest != expected_after:
            raise ValueError(
                "projected cache-after manifest does not match cache-before plus current"
            )
        object.__setattr__(self, "cache_before_manifest", cache_before_manifest)
        object.__setattr__(
            self,
            "projected_cache_after_manifest",
            projected_cache_after_manifest,
        )
        if self.memory_budget_bytes is not None:
            _require_plain_int(
                self.memory_budget_bytes,
                name="memory_budget_bytes",
                minimum=1,
            )
        _require_plain_int(
            self.model_residency_bytes,
            name="model_residency_bytes",
            minimum=0,
        )
        buffers = tuple(self.logical_buffers)
        if any(not isinstance(buffer, HDLogicalBuffer) for buffer in buffers):
            raise TypeError("logical_buffers must contain HDLogicalBuffer values")
        output_buffers = tuple(buffer for buffer in buffers if buffer.name == "output")
        if (
            len(output_buffers) != 1
            or output_buffers[0].dtype not in _INPUT_DTYPE_NAMES
        ):
            raise ValueError("plan must contain one supported floating output buffer")
        expected_buffers = _expected_logical_buffers(
            geometry=self.geometry,
            token_block=self.token_block,
            feature_wave_blocks=self.feature_wave_blocks,
            input_dtype=self.input_dtype,
            output_dtype=output_buffers[0].dtype,
            pair_layout_id=self.pair_layout_id,
            device=self.device,
            projected_cache_after_manifest=projected_cache_after_manifest,
            requested_gradient_mask=self.requested_gradient_mask,
            precision=self.precision,
            result_contract=self.result_contract,
            key_feature_impl=self.key_feature_impl,
            key_fold_impl=self.key_fold_impl,
            query_feature_impl=self.query_feature_impl,
            query_fold_impl=self.query_fold_impl,
            query_fold_input=self.query_fold_input,
            query_gradient_flow=self.query_gradient_flow,
            query_producer_fold_strategy=self.query_producer_fold_strategy,
            query_producer_partition_count=self.query_producer_partition_count,
            query_consumer_stages=self.query_consumer_stages,
            backward_schedule=self.backward_schedule,
            gradient_staging=self.gradient_staging,
            backward_normalize_impl=self.backward_normalize_impl,
            save_local_score=self.save_local_score,
            key_retention=self.key_retention,
            physical_feature_dimension=self.physical_feature_dimension,
        )
        if buffers != expected_buffers:
            raise ValueError(
                "logical_buffers must equal the canonical logical buffer table"
            )
        object.__setattr__(self, "logical_buffers", expected_buffers)
        expected_summaries = _byte_summaries(
            expected_buffers,
            self.requested_gradient_mask,
            cache_before_manifest,
        )
        actual_summaries = {name: getattr(self, name) for name in expected_summaries}
        if actual_summaries != expected_summaries:
            raise ValueError(
                "plan byte summaries do not match logical buffer lifetimes"
            )
        if (
            self.projected_live_bytes
            != self.invocation_peak_bytes + self.model_residency_bytes
        ):
            raise ValueError("projected live bytes must include model residency")
        if (
            self.memory_budget_bytes is not None
            and self.projected_live_bytes > self.memory_budget_bytes
        ):
            raise ValueError(
                "memory budget is below the exact invocation peak plus model residency"
            )
        expected_plan_id = _sha256(self.payload_without_plan_id())
        if self.plan_id and self.plan_id != expected_plan_id:
            raise ValueError("HD Block-GEMM plan id does not match its stable payload")
        object.__setattr__(self, "plan_id", expected_plan_id)

    @property
    def batch_size(self) -> int:
        return self.geometry[0]

    @property
    def query_heads(self) -> int:
        return self.geometry[1]

    @property
    def key_value_heads(self) -> int:
        return self.geometry[2]

    @property
    def sequence_length(self) -> int:
        return self.geometry[3]

    @property
    def head_dimension(self) -> int:
        return self.geometry[4]

    @property
    def value_dimension(self) -> int:
        return self.geometry[5]

    @property
    def output_dtype(self) -> str:
        """Normalized output dtype encoded by the canonical output buffer."""
        return self.buffer("output").dtype

    @property
    def BT(self) -> int:
        """Compatibility spelling used by physical-path evidence."""
        return self.token_block

    @property
    def wave(self) -> int:
        """Compatibility spelling used by physical-path evidence."""
        return self.feature_wave_blocks

    @property
    def query_consumer_fusion(self) -> str:
        """Legacy read-only spelling retained for private compatibility reads."""
        return legacy_query_consumer_fusion_for_stages(self.query_consumer_stages)

    @property
    def saved_local_score_bytes(self) -> int:
        """Bytes retained from forward solely for the local dV contraction."""
        try:
            return self.buffer("saved_local_score_tc").nbytes
        except KeyError:
            return 0

    @property
    def forward_phi_k_cache_bytes(self) -> int:
        """Bytes retained only between the two forward passes."""
        try:
            return self.buffer("forward_phi_k_cache").nbytes
        except KeyError:
            return 0

    @property
    def saved_phi_k_bytes(self) -> int:
        """Bytes retained from forward for backward key-feature consumers."""
        try:
            return self.buffer("saved_phi_k_tc").nbytes
        except KeyError:
            return 0

    @property
    def gradient_staging_bytes(self) -> int:
        """Bytes in the optional invocation-local full BF16 gradient cache."""
        try:
            return self.buffer("backward_g_full_bf16").nbytes
        except KeyError:
            return 0

    def buffer(self, name: str) -> HDLogicalBuffer:
        for buffer in self.logical_buffers:
            if buffer.name == name:
                return buffer
        raise KeyError(name)

    def payload_without_plan_id(self) -> dict[str, Any]:
        payload = {
            "schema": "hd_parallel_block_plan_v4",
            "physical_path": self.physical_path,
            "token_block": self.token_block,
            "feature_wave_blocks": self.feature_wave_blocks,
            "feature_storage": self.feature_storage,
            "feature_padding": self.feature_padding,
            "result_contract": self.result_contract,
            "key_feature_impl": self.key_feature_impl,
            "key_fold_impl": self.key_fold_impl,
            "query_feature_impl": self.query_feature_impl,
            "query_feature_token_tile": self.query_feature_token_tile,
            "query_fold_impl": self.query_fold_impl,
            "query_fold_token_tile": self.query_fold_token_tile,
            "query_fold_input": self.query_fold_input,
            "query_gradient_flow": self.query_gradient_flow,
            "query_producer_fold_strategy": self.query_producer_fold_strategy,
            "query_producer_partition_count": self.query_producer_partition_count,
            "query_consumer_stages": list(self.query_consumer_stages),
            "backward_schedule": self.backward_schedule,
            "forward_normalize_impl": self.forward_normalize_impl,
            "backward_normalize_impl": self.backward_normalize_impl,
            "kv_cross_impl": self.kv_cross_impl,
            "gradient_staging": self.gradient_staging,
            "save_local_score": True,
            "requested_key_retention": self.requested_key_retention,
            "key_retention": self.key_retention,
            "requested_precision": self.requested_precision,
            "precision": self.precision,
            "precision_fallback_reason": self.precision_fallback_reason,
            "contraction_backend": self.contraction_backend_identity.to_dict(),
            "requested_gradient_mask": list(self.requested_gradient_mask),
            "geometry": {
                "batch_size": self.batch_size,
                "query_heads": self.query_heads,
                "key_value_heads": self.key_value_heads,
                "sequence_length": self.sequence_length,
                "head_dimension": self.head_dimension,
                "value_dimension": self.value_dimension,
                "gqa_ratio": self.gqa_ratio,
                "pair_count": self.pair_count,
                "feature_dimension": self.feature_dimension,
                "physical_feature_dimension": self.physical_feature_dimension,
                "augmented_value_dimension": self.augmented_value_dimension,
                "physical_augmented_value_dimension": self.physical_augmented_value_dimension,
                "number_blocks": self.number_blocks,
                "input_dtype": self.input_dtype,
                "input_layout": self.input_layout,
                "input_strides": [list(strides) for strides in self.input_strides],
                "device_type": self.device_type,
                "device": self.device,
            },
            "pair_layout_id": self.pair_layout_id,
            "cache_before_manifest": [
                entry.to_dict() for entry in self.cache_before_manifest
            ],
            "projected_cache_after_manifest": [
                entry.to_dict() for entry in self.projected_cache_after_manifest
            ],
            "memory_budget_bytes": self.memory_budget_bytes,
            "model_residency_bytes": self.model_residency_bytes,
            "execution_stages": list(self.execution_stages),
            "logical_buffers": [buffer.to_dict() for buffer in self.logical_buffers],
            "byte_summaries": {
                "forward_block_totals_bytes": self.forward_block_totals_bytes,
                "saved_carry_bytes": self.saved_carry_bytes,
                "output_bytes": self.output_bytes,
                "largest_feature_wave_bytes": self.largest_feature_wave_bytes,
                "local_score_bytes": self.local_score_bytes,
                "forward_ephemeral_peak_bytes": self.forward_ephemeral_peak_bytes,
                "saved_tensor_bytes": self.saved_tensor_bytes,
                "saved_coefficient_bytes": self.saved_coefficient_bytes,
                "saved_activation_bytes": self.saved_activation_bytes,
                "cache_before_residency_bytes": (self.cache_before_residency_bytes),
                "core_metadata_residency_bytes": self.core_metadata_residency_bytes,
                "pair_metadata_materialization_bytes": (
                    self.pair_metadata_materialization_bytes
                ),
                "backward_right_scan_bytes": self.backward_right_scan_bytes,
                "backward_scratch_peak_bytes": self.backward_scratch_peak_bytes,
                "invocation_peak_bytes": self.invocation_peak_bytes,
                "projected_live_bytes": self.projected_live_bytes,
            },
        }

        # Default omission is part of the v4 wire format, not a config default.
        for name, default in (
            ("feature_padding", "none"),
            ("query_feature_token_tile", 1),
            ("query_fold_token_tile", 1),
            ("query_fold_input", "staged_fp32"),
            ("query_producer_partition_count", None),
            ("forward_normalize_impl", "torch"),
            ("backward_normalize_impl", "torch"),
            ("kv_cross_impl", "split"),
            ("gradient_staging", "per_wave"),
            ("key_retention", "none"),
        ):
            if payload[name] == default:
                del payload[name]
        if not self.save_local_score:
            del payload["save_local_score"]
        if self.requested_key_retention != "backward":
            del payload["requested_key_retention"]
        if self.feature_padding == "none":
            del payload["geometry"]["physical_feature_dimension"]
            del payload["geometry"]["physical_augmented_value_dimension"]
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload_without_plan_id(), "plan_id": self.plan_id}

    def to_json(self) -> str:
        return _stable_json(self.to_dict())


# Keep existing serialized plans importable through their historical public path.
HDParallelBlockPlan.__module__ = "oal_attention.hd_block_gemm"

__all__ = ("HDParallelBlockPlan",)
