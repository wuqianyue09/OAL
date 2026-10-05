"""Strict private launch facades for HD Block-GEMM key operators."""

from __future__ import annotations

import torch

from .hd_block_gemm_plan import HDParallelBlockPlan
from .triton import hd_block_gemm_key_kernels


def _reference_key_features(
    key: torch.Tensor,
    pair_rows: torch.Tensor,
    pair_columns: torch.Tensor,
) -> torch.Tensor:
    """Return the canonical key basis using FP32 products and BF16 storage."""
    key_fp32 = key.float()
    return torch.cat(
        (
            torch.ones((*key.shape[:-1], 1), dtype=torch.float32, device=key.device),
            key_fp32,
            key_fp32.index_select(-1, pair_rows)
            * key_fp32.index_select(-1, pair_columns),
        ),
        dim=-1,
    ).to(torch.bfloat16)


def _reference_key_feature_gradient(
    key: torch.Tensor,
    d_phi_key: torch.Tensor,
    *,
    pair_rows: torch.Tensor,
    pair_columns: torch.Tensor,
) -> torch.Tensor:
    """Fold the canonical key basis with explicit endpoint reductions."""
    head_dimension = key.shape[-1]
    pair_start = 1 + head_dimension
    output = d_phi_key[..., 1:pair_start].clone()
    pair_count = pair_rows.numel()
    pair_gradient = d_phi_key[..., pair_start : pair_start + pair_count]
    row_source = pair_gradient * key.index_select(-1, pair_columns)
    column_source = pair_gradient * key.index_select(-1, pair_rows)
    row_index = pair_rows.view(*([1] * (key.ndim - 1)), -1).expand_as(row_source)
    column_index = pair_columns.view(*([1] * (key.ndim - 1)), -1).expand_as(
        column_source
    )
    output.scatter_add_(-1, row_index, row_source)
    output.scatter_add_(-1, column_index, column_source)
    return output


def _require_key_feature_plan(plan: object) -> HDParallelBlockPlan:
    if not isinstance(plan, HDParallelBlockPlan):
        raise TypeError("plan must be an HDParallelBlockPlan")
    if plan.key_feature_impl != "triton_materialized":
        raise ValueError(
            "Triton key features require key_feature_impl=triton_materialized"
        )
    if (
        plan.precision != "bf16_tensorcore"
        or plan.head_dimension != 64
        or plan.result_contract != "full_aux"
    ):
        raise ValueError(
            "Triton key features require bf16_tensorcore, D=64, and full_aux"
        )
    return plan


def _require_key_fold_plan(plan: object) -> HDParallelBlockPlan:
    if not isinstance(plan, HDParallelBlockPlan):
        raise TypeError("plan must be an HDParallelBlockPlan")
    if plan.key_fold_impl != "triton_materialized":
        raise ValueError("Triton key fold requires key_fold_impl=triton_materialized")
    if (
        plan.precision != "bf16_tensorcore"
        or plan.head_dimension != 64
        or plan.result_contract != "full_aux"
    ):
        raise ValueError("Triton key fold requires bf16_tensorcore, D=64, and full_aux")
    if not plan.requested_gradient_mask[1]:
        raise ValueError("Triton key fold requires the K gradient")
    return plan


def _require_tensor(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    dense_last_two: bool = False,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.dtype != dtype:
        raise ValueError(f"{name} must use {dtype}")
    if value.device != device:
        raise ValueError(f"{name} must match the plan device")
    if value.layout != torch.strided:
        raise ValueError(f"{name} must use strided layout")
    if dense_last_two:
        if value.stride(-1) != 1 or value.stride(-2) != shape[-1]:
            raise ValueError(f"{name} must have a dense feature axis and token stride")
    elif not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return value


def build_triton_key_features(
    key: object,
    output: object,
    *,
    pair_rows: object,
    pair_columns: object,
    plan: object,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Materialize one plan-bound key wave without generic pair scratch."""
    plan = _require_key_feature_plan(plan)
    if not isinstance(key, torch.Tensor):
        raise TypeError("K must be a tensor")
    if key.ndim != 4:
        raise ValueError("K must have shape [B,Hkv,T,D]")
    if not key.is_cuda:
        raise RuntimeError("Triton key features require CUDA K")
    device = torch.device(plan.device)
    if key.device != device:
        raise ValueError("K device does not match the plan")
    token_count = key.shape[2]
    if key.shape != (
        plan.batch_size,
        plan.key_value_heads,
        token_count,
        plan.head_dimension,
    ):
        raise ValueError("K shape does not match the D=64 feature plan")
    if token_count <= 0 or token_count > plan.feature_wave_blocks * plan.token_block:
        raise ValueError("K token count exceeds the planned feature wave")
    if key.dtype != torch.bfloat16:
        raise ValueError("Triton key features require BF16 K")
    if (
        key.layout != torch.strided
        or key.stride(-1) != 1
        or key.stride(-2) != plan.head_dimension
    ):
        raise ValueError("K must have a dense feature axis and token stride")

    output = _require_tensor(
        output,
        name="output",
        shape=(
            plan.batch_size,
            plan.key_value_heads,
            token_count,
            plan.physical_feature_dimension,
        ),
        dtype=torch.bfloat16,
        device=device,
        dense_last_two=True,
    )
    pair_rows = _require_tensor(
        pair_rows,
        name="pair_rows",
        shape=(plan.pair_count,),
        dtype=torch.int64,
        device=device,
    )
    pair_columns = _require_tensor(
        pair_columns,
        name="pair_columns",
        shape=(plan.pair_count,),
        dtype=torch.int64,
        device=device,
    )
    if key.untyped_storage().data_ptr() == output.untyped_storage().data_ptr():
        raise ValueError("K and output must not overlap")
    if not hd_block_gemm_key_kernels.triton_is_available():
        raise RuntimeError("Triton key features require Triton")
    current_stream = torch.cuda.current_stream(device)
    if stream is not None:
        if not isinstance(stream, torch.cuda.Stream) or stream.device != device:
            raise ValueError("Triton key feature stream does not match the plan device")
        if stream.cuda_stream != current_stream.cuda_stream:
            raise RuntimeError("Triton key features must launch on the current stream")
    if plan.feature_padding != "none":
        output[..., plan.feature_dimension :].zero_()
    hd_block_gemm_key_kernels.build_key_features(
        key,
        output,
        pair_rows=pair_rows,
        pair_columns=pair_columns,
        head_dimension=plan.head_dimension,
        pair_count=plan.pair_count,
        feature_dimension=plan.feature_dimension,
    )
    return output


def fold_triton_key_feature_gradient(
    key: object,
    d_phi_key: object,
    output: object,
    *,
    plan: object,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Fold one plan-bound key-feature gradient without scatter scratch."""
    plan = _require_key_fold_plan(plan)
    if not isinstance(key, torch.Tensor):
        raise TypeError("K must be a tensor")
    if key.ndim != 4:
        raise ValueError("K must have shape [B,Hkv,T,D]")
    if not key.is_cuda:
        raise RuntimeError("Triton key fold requires CUDA K")
    device = torch.device(plan.device)
    if key.device != device:
        raise ValueError("K device does not match the plan")
    token_count = key.shape[2]
    key_shape = (
        plan.batch_size,
        plan.key_value_heads,
        token_count,
        plan.head_dimension,
    )
    if tuple(key.shape) != key_shape:
        raise ValueError("K shape does not match the D=64 key-fold plan")
    if token_count <= 0 or token_count > plan.feature_wave_blocks * plan.token_block:
        raise ValueError("K token count exceeds the planned feature wave")
    if key.dtype != torch.bfloat16:
        raise ValueError("Triton key fold requires BF16 K")
    if (
        key.layout != torch.strided
        or key.stride(-1) != 1
        or key.stride(-2) != plan.head_dimension
    ):
        raise ValueError("K must have a dense feature axis and token stride")

    d_phi_key = _require_tensor(
        d_phi_key,
        name="dPhiK",
        shape=(
            plan.batch_size,
            plan.key_value_heads,
            token_count,
            plan.physical_feature_dimension,
        ),
        dtype=torch.float32,
        device=device,
        dense_last_two=True,
    )
    output = _require_tensor(
        output,
        name="output",
        shape=key_shape,
        dtype=torch.float32,
        device=device,
        dense_last_two=True,
    )
    storage_pointers = {
        key.untyped_storage().data_ptr(),
        d_phi_key.untyped_storage().data_ptr(),
        output.untyped_storage().data_ptr(),
    }
    if len(storage_pointers) != 3:
        raise ValueError("K, dPhiK, and output must not overlap")
    if not hd_block_gemm_key_kernels.triton_is_available():
        raise RuntimeError("Triton key fold requires Triton")
    current_stream = torch.cuda.current_stream(device)
    if stream is not None:
        if not isinstance(stream, torch.cuda.Stream) or stream.device != device:
            raise ValueError("Triton key fold stream does not match the plan device")
        if stream.cuda_stream != current_stream.cuda_stream:
            raise RuntimeError("Triton key fold must launch on the current stream")
    hd_block_gemm_key_kernels.fold_key_feature_gradient(
        key,
        d_phi_key,
        output,
        head_dimension=plan.head_dimension,
        pair_count=plan.pair_count,
        feature_dimension=plan.feature_dimension,
    )
    return output


__all__ = ("build_triton_key_features", "fold_triton_key_feature_gradient")
