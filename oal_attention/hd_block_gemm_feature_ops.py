"""Strict private launch facade for staged HD Block-GEMM feature operators."""

from __future__ import annotations

import torch

from .hd_block_gemm_plan import HDParallelBlockPlan
from .triton import hd_block_gemm_feature_kernels


def _require_feature_plan(plan: object) -> HDParallelBlockPlan:
    if not isinstance(plan, HDParallelBlockPlan):
        raise TypeError("plan must be an HDParallelBlockPlan")
    if plan.query_feature_impl != "triton_materialized":
        raise ValueError(
            "Triton query features require query_feature_impl=triton_materialized"
        )
    if (
        plan.query_gradient_flow != "materialized"
        or plan.query_producer_fold_strategy != "none"
    ):
        raise ValueError("query_gradient_flow: OP-3 exploration is retired")
    if plan.query_consumer_stages:
        raise ValueError("Triton query features require no query consumer stages")
    if (
        plan.precision != "bf16_tensorcore"
        or plan.head_dimension != 64
        or plan.result_contract != "full_aux"
    ):
        raise ValueError(
            "Triton query features require bf16_tensorcore, D=64, and full_aux"
        )
    return plan


def _require_fold_plan(plan: object) -> HDParallelBlockPlan:
    if not isinstance(plan, HDParallelBlockPlan):
        raise TypeError("plan must be an HDParallelBlockPlan")
    if plan.query_fold_impl != "triton_materialized":
        raise ValueError(
            "Triton query fold requires query_fold_impl=triton_materialized"
        )
    if plan.query_gradient_flow != "materialized":
        raise ValueError("Triton query fold requires query_gradient_flow=materialized")
    if plan.query_consumer_stages:
        raise ValueError("Triton query fold requires no query consumer stages")
    if (
        plan.precision != "bf16_tensorcore"
        or plan.head_dimension != 64
        or plan.result_contract != "full_aux"
    ):
        raise ValueError(
            "Triton query fold requires bf16_tensorcore, D=64, and full_aux"
        )
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
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.dtype != dtype:
        raise ValueError(f"{name} must use {dtype}")
    if value.device != device:
        raise ValueError(f"{name} must share the query CUDA device")
    if value.layout != torch.strided:
        raise ValueError(f"{name} must use strided layout")
    if dense_last_two:
        if value.stride(-1) != 1 or value.stride(-2) != shape[-1]:
            raise ValueError(f"{name} must have a dense feature axis and token stride")
    elif not value.is_contiguous():
        raise ValueError(f"{name} must be a contiguous strided tensor")
    return value


def build_triton_query_features(
    q: object,
    output: object,
    *,
    a: object,
    b: object,
    c: object,
    scale: object,
    pair_rows: object,
    pair_columns: object,
    pair_multiplicity: object,
    plan: object,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Materialize one plan-bound query wave without temporary workspaces."""

    plan = _require_feature_plan(plan)
    if not isinstance(q, torch.Tensor):
        raise TypeError("Q must be a tensor")
    if q.ndim != 4:
        raise ValueError("Q must have shape [B,Hq,T,D]")
    if not q.is_cuda:
        raise RuntimeError("Triton query features require CUDA Q")
    device = torch.device(plan.device)
    if q.device != device:
        raise ValueError("Q device does not match the plan")
    token_count = q.shape[2]
    if q.shape != (plan.batch_size, plan.query_heads, token_count, 64):
        raise ValueError("Q shape does not match the D=64 feature plan")
    if token_count <= 0 or token_count > plan.feature_wave_blocks * plan.token_block:
        raise ValueError("Q token count exceeds the planned feature wave")
    if q.dtype != torch.bfloat16:
        raise ValueError("Triton query features require BF16 Q")
    if q.layout != torch.strided or q.stride(-1) != 1 or q.stride(-2) != 64:
        raise ValueError("Q must have a dense feature axis and token stride")

    output = _require_tensor(
        output,
        name="output",
        shape=(
            plan.batch_size,
            plan.query_heads,
            token_count,
            plan.physical_feature_dimension,
        ),
        dtype=torch.bfloat16,
        device=device,
        dense_last_two=True,
    )
    a = _require_tensor(
        a,
        name="A",
        shape=(plan.query_heads,),
        dtype=torch.float32,
        device=device,
    )
    b = _require_tensor(
        b,
        name="B",
        shape=(plan.query_heads, 64),
        dtype=torch.float32,
        device=device,
    )
    c = _require_tensor(
        c,
        name="C",
        shape=(plan.query_heads, plan.pair_count),
        dtype=torch.float32,
        device=device,
    )
    scale = _require_tensor(
        scale,
        name="scale",
        shape=(1,),
        dtype=torch.float32,
        device=device,
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
    pair_multiplicity = _require_tensor(
        pair_multiplicity,
        name="pair_multiplicity",
        shape=(plan.pair_count,),
        dtype=torch.int64,
        device=device,
    )
    if not hd_block_gemm_feature_kernels.triton_is_available():
        raise RuntimeError("Triton query features require an installed Triton package")
    current_stream = torch.cuda.current_stream(device)
    if stream is not None:
        if not isinstance(stream, torch.cuda.Stream) or stream.device != device:
            raise ValueError(
                "Triton query feature stream does not match the plan device"
            )
        if stream.cuda_stream != current_stream.cuda_stream:
            raise RuntimeError(
                "Triton query features must launch on the current stream"
            )
    if plan.feature_padding != "none":
        output[..., plan.feature_dimension :].zero_()
    hd_block_gemm_feature_kernels.build_query_features(
        q,
        output,
        a=a,
        b=b,
        c=c,
        scale=scale,
        pair_rows=pair_rows,
        pair_columns=pair_columns,
        pair_multiplicity=pair_multiplicity,
        token_tile=plan.query_feature_token_tile,
    )
    return output


def fold_triton_query_feature_gradient(
    q: object,
    d_phi_q: object,
    d_q: object,
    *,
    b: object,
    c: object,
    scale: object,
    pair_rows: object,
    pair_columns: object,
    pair_multiplicity: object,
    a_block_partials: object,
    b_block_partials: object,
    c_block_partials: object,
    block_start: object,
    q_is_raw: bool = False,
    valid_tokens: int | None = None,
    plan: object,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Fold one planned FP32 query wave without generic pair scratch.

    ``q`` is either the existing scaled FP32 work or the raw BF16 wave selected
    by the private plan.  Raw mode performs the input scale inside each consumer
    while retaining the separate dQ chain-rule scale.
    """

    plan = _require_fold_plan(plan)
    hd_block_gemm_feature_kernels._query_fold_token_tile(
        plan.token_block
    )  # noqa: SLF001
    device = torch.device(plan.device)
    need_q, _, _, need_a, need_b, need_c = plan.requested_gradient_mask
    needs_input_q = need_q or need_b or need_c
    coefficient_requested = need_a or need_b or need_c

    if not isinstance(d_phi_q, torch.Tensor):
        raise TypeError("dPhiQ must be a tensor")
    if d_phi_q.ndim != 5:
        raise ValueError("dPhiQ must have shape [B,Hq,WB,BT,F]")
    wave_blocks = d_phi_q.shape[2]
    wave_capacity = wave_blocks * plan.token_block
    expected_d_phi_shape = (
        plan.batch_size,
        plan.query_heads,
        wave_blocks,
        plan.token_block,
        plan.physical_feature_dimension,
    )
    if d_phi_q.shape != expected_d_phi_shape or wave_blocks <= 0:
        raise ValueError("dPhiQ shape does not match the planned query wave")
    if wave_blocks > plan.feature_wave_blocks:
        raise ValueError("dPhiQ exceeds the planned feature wave capacity")
    if not d_phi_q.is_cuda:
        raise RuntimeError("Triton query fold requires CUDA dPhiQ")
    if d_phi_q.device != device:
        raise ValueError("dPhiQ device does not match the plan")
    if d_phi_q.dtype != torch.float32:
        raise ValueError("Triton query fold requires FP32 dPhiQ")
    if (
        d_phi_q.layout != torch.strided
        or d_phi_q.stride(-1) != 1
        or d_phi_q.stride(-2) != plan.physical_feature_dimension
    ):
        raise ValueError("dPhiQ must have a dense feature axis and token stride")

    expected_raw = getattr(plan, "query_fold_input", "staged_fp32") == "raw"
    if q_is_raw != expected_raw:
        raise ValueError("live query fold input does not match the plan")
    if valid_tokens is None:
        valid_tokens = wave_capacity
    if (
        not isinstance(valid_tokens, int)
        or isinstance(valid_tokens, bool)
        or valid_tokens <= 0
        or valid_tokens > wave_capacity
    ):
        raise ValueError("valid_tokens is outside the planned query wave")

    def require_query_wave(
        value: object,
        *,
        name: str,
        raw_input: bool = False,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        expected_shape = (
            (plan.batch_size, plan.query_heads, valid_tokens, plan.head_dimension)
            if raw_input
            else (
                plan.batch_size,
                plan.query_heads,
                wave_blocks,
                plan.token_block,
                plan.head_dimension,
            )
        )
        if value.shape != expected_shape:
            raise ValueError(f"{name} shape does not match the planned query wave")
        if not value.is_cuda:
            raise RuntimeError(f"Triton query fold requires CUDA {name}")
        if value.device != device:
            raise ValueError(f"{name} device does not match the plan")
        expected_dtype = torch.bfloat16 if raw_input else torch.float32
        if value.dtype != expected_dtype:
            raise ValueError(f"Triton query fold requires {expected_dtype} {name}")
        if (
            value.layout != torch.strided
            or value.stride(-1) != 1
            or value.stride(-2) != plan.head_dimension
        ):
            raise ValueError(f"{name} must have a dense feature axis and token stride")
        return value

    q_tensor = (
        require_query_wave(q, name="Q", raw_input=q_is_raw) if needs_input_q else None
    )
    if not needs_input_q and q is not None:
        raise ValueError("Q is not required by the requested query gradients")
    d_q_tensor = require_query_wave(d_q, name="dQ") if need_q else None
    if not need_q and d_q is not None:
        raise ValueError("dQ is not required by the requested query gradients")

    b_tensor = None
    c_tensor = None
    scale_tensor = None
    pair_rows_tensor = None
    pair_columns_tensor = None
    pair_multiplicity_tensor = None
    if needs_input_q:
        scale_tensor = _require_tensor(
            scale,
            name="scale",
            shape=(1,),
            dtype=torch.float32,
            device=device,
        )
    if need_q:
        b_tensor = _require_tensor(
            b,
            name="B",
            shape=(plan.query_heads, plan.head_dimension),
            dtype=torch.float32,
            device=device,
        )
        c_tensor = _require_tensor(
            c,
            name="C",
            shape=(plan.query_heads, plan.pair_count),
            dtype=torch.float32,
            device=device,
        )
    if need_c:
        pair_rows_tensor = _require_tensor(
            pair_rows,
            name="pair_rows",
            shape=(plan.pair_count,),
            dtype=torch.int64,
            device=device,
        )
        pair_columns_tensor = _require_tensor(
            pair_columns,
            name="pair_columns",
            shape=(plan.pair_count,),
            dtype=torch.int64,
            device=device,
        )
        pair_multiplicity_tensor = _require_tensor(
            pair_multiplicity,
            name="pair_multiplicity",
            shape=(plan.pair_count,),
            dtype=torch.int64,
            device=device,
        )

    if coefficient_requested:
        if not isinstance(block_start, int) or isinstance(block_start, bool):
            raise TypeError("block_start must be an integer for coefficient gradients")
        if block_start < 0 or block_start + wave_blocks > plan.number_blocks:
            raise ValueError("coefficient gradient block range is out of bounds")
    elif block_start is not None:
        raise ValueError("block_start is only valid for coefficient gradients")

    a_partials_tensor = (
        _require_tensor(
            a_block_partials,
            name="dA block partials",
            shape=(plan.batch_size, plan.number_blocks, plan.query_heads),
            dtype=torch.float32,
            device=device,
        )
        if need_a
        else None
    )
    b_partials_tensor = (
        _require_tensor(
            b_block_partials,
            name="dB block partials",
            shape=(
                plan.batch_size,
                plan.number_blocks,
                plan.query_heads,
                plan.head_dimension,
            ),
            dtype=torch.float32,
            device=device,
        )
        if need_b
        else None
    )
    c_partials_tensor = (
        _require_tensor(
            c_block_partials,
            name="dC block partials",
            shape=(
                plan.batch_size,
                plan.number_blocks,
                plan.query_heads,
                plan.pair_count,
            ),
            dtype=torch.float32,
            device=device,
        )
        if need_c
        else None
    )
    if not need_a and a_block_partials is not None:
        raise ValueError("dA block partials are not required by the plan")
    if not need_b and b_block_partials is not None:
        raise ValueError("dB block partials are not required by the plan")
    if not need_c and c_block_partials is not None:
        raise ValueError("dC block partials are not required by the plan")

    if not hd_block_gemm_feature_kernels.triton_is_available():
        raise RuntimeError("Triton query fold requires an installed Triton package")
    current_stream = torch.cuda.current_stream(device)
    if stream is not None:
        if not isinstance(stream, torch.cuda.Stream) or stream.device != device:
            raise ValueError("Triton query fold stream does not match the plan device")
        if stream.cuda_stream != current_stream.cuda_stream:
            raise RuntimeError("Triton query fold must launch on the current stream")
    hd_block_gemm_feature_kernels.fold_query_feature_gradient(
        q_tensor,
        d_phi_q[..., : plan.feature_dimension],
        d_q_tensor,
        b=b_tensor,
        c=c_tensor,
        scale=scale_tensor,
        pair_rows=pair_rows_tensor,
        pair_columns=pair_columns_tensor,
        pair_multiplicity=pair_multiplicity_tensor,
        a_block_partials=a_partials_tensor,
        b_block_partials=b_partials_tensor,
        c_block_partials=c_partials_tensor,
        block_start=block_start,
        q_is_raw=q_is_raw,
        valid_tokens=valid_tokens,
        dq_token_tile=getattr(plan, "query_fold_token_tile", 1),
    )


__all__: tuple[str, ...] = ()
