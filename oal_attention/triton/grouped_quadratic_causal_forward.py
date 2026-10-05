"""KV-owned streaming causal forward for grouped quadratic attention."""

from __future__ import annotations

import torch

from .grouped_quadratic_causal_common import (
    DIAGNOSTIC_MACRO_TOKEN_BLOCK,
    PHYSICAL_PATH_IDENTIFIER,
    GroupedCausalForwardDiagnosticWitness,
    GroupedCausalForwardTokenWitness,
    canonical_group_indices,
    canonical_packed_pair_metadata,
    require_cpu_diagnostic_geometry,
)
from . import grouped_quadratic_causal_forward_kernels as triton_forward_kernels


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    *,
    scale: float,
) -> tuple[torch.Tensor, int, int, int, int]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Q/K/V must be rank-4 tensors")
    batch_size, query_heads, token_count, head_dimension = q.shape
    if k.shape[0] != batch_size or v.shape[0] != batch_size:
        raise ValueError("Q/K/V batch dimensions must agree")
    if k.shape[2] != token_count or v.shape[2] != token_count:
        raise ValueError("Q/K/V token dimensions must agree")
    if k.shape[-1] != head_dimension:
        raise ValueError("Q and K head dimensions must agree")
    key_value_heads = k.shape[1]
    if query_heads % key_value_heads != 0:
        raise ValueError("query heads must be divisible by KV heads")
    value_dimension = v.shape[-1]
    gmax = linear.shape[-1]
    if (
        constant.shape != (query_heads,)
        or linear.shape != (query_heads, gmax)
        or quadratic.shape != (query_heads, gmax, gmax)
    ):
        raise ValueError("expanded A/B/C shapes must agree with Q heads and Gmax")
    if not all(
        tensor.device == q.device
        for tensor in (k, v, dim_groups, constant, linear, quadratic)
    ):
        raise ValueError("grouped causal forward tensors must share one device")
    if not isinstance(scale, (int, float)) or isinstance(scale, bool):
        raise TypeError("scale must be a scalar")
    groups = canonical_group_indices(
        dim_groups,
        query_heads=query_heads,
        head_dimension=head_dimension,
        gmax=gmax,
    )
    return groups, batch_size, key_value_heads, value_dimension, head_dimension


def grouped_causal_streaming_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return causal output/numerator/denominator without prefix histories.

    hadamard_h012_packed_diag_v1
    """
    groups, batch_size, key_value_heads, value_dimension, head_dimension = (
        _validate_inputs(q, k, v, dim_groups, constant, linear, quadratic, scale=scale)
    )
    if (
        q.is_cuda
        and q.dtype in {torch.float16, torch.bfloat16}
        and triton_forward_kernels.triton_is_available()
    ):
        kernel_plan = triton_forward_kernels.build_forward_kernel_plan(
            q, k, v, dim_groups, constant, linear, quadratic
        )
        return triton_forward_kernels.grouped_causal_triton_forward(
            q,
            k,
            v,
            dim_groups,
            constant,
            linear,
            quadratic,
            scale=scale,
            kernel_plan=kernel_plan,
        )
    query_heads = q.shape[1]
    token_count = q.shape[2]
    pair_rows, pair_columns = torch.tril_indices(
        head_dimension, head_dimension, device=q.device, dtype=torch.int64
    )
    pair_count = pair_rows.numel()
    multiplicity = torch.where(
        pair_rows == pair_columns,
        torch.ones(pair_count, device=q.device, dtype=torch.float32),
        torch.full((pair_count,), 2.0, device=q.device, dtype=torch.float32),
    )
    query_to_key_value = torch.arange(query_heads, device=q.device) // (
        query_heads // key_value_heads
    )
    q_work = q.float()
    k_work = k.float()
    v_work = v.float()
    constant_work = constant.float()
    linear_work = linear.float()
    quadratic_work = quadratic.float()
    linear_by_dimension = linear_work.gather(1, groups)
    pair_row_groups = groups[:, pair_rows]
    pair_column_groups = groups[:, pair_columns]
    quadratic_by_pair = quadratic_work[
        torch.arange(query_heads, device=q.device)[:, None],
        pair_row_groups,
        pair_column_groups,
    ]
    h0 = torch.zeros(
        (batch_size, key_value_heads, value_dimension + 1),
        device=q.device,
        dtype=torch.float32,
    )
    h1 = torch.zeros(
        (batch_size, key_value_heads, head_dimension, value_dimension + 1),
        device=q.device,
        dtype=torch.float32,
    )
    h2 = torch.zeros(
        (batch_size, key_value_heads, pair_count, value_dimension + 1),
        device=q.device,
        dtype=torch.float32,
    )
    numerator = torch.empty(
        (batch_size, query_heads, token_count, value_dimension),
        device=q.device,
        dtype=torch.float32,
    )
    denominator = torch.empty(
        (batch_size, query_heads, token_count, 1),
        device=q.device,
        dtype=torch.float32,
    )
    for token in range(token_count):
        augmented_value = torch.empty(
            (batch_size, key_value_heads, value_dimension + 1),
            device=q.device,
            dtype=torch.float32,
        )
        augmented_value[..., :value_dimension] = v_work[:, :, token, :]
        augmented_value[..., value_dimension] = 1.0
        key_token = k_work[:, :, token, :]
        h0.add_(augmented_value)
        h1.add_(key_token.unsqueeze(-1) * augmented_value.unsqueeze(-2))
        key_pair = key_token[..., pair_rows] * key_token[..., pair_columns]
        h2.add_(key_pair.unsqueeze(-1) * augmented_value.unsqueeze(-2))
        query_scaled = q_work[:, :, token, :] * float(scale)
        h0_query = h0.index_select(1, query_to_key_value)
        h1_query = h1.index_select(1, query_to_key_value)
        h2_query = h2.index_select(1, query_to_key_value)
        augmented = constant_work[None, :, None] * h0_query
        augmented.add_(
            (
                query_scaled[:, :, :, None]
                * h1_query
                * linear_by_dimension[None, :, :, None]
            ).sum(dim=2)
        )
        pair_query = query_scaled[..., pair_rows] * query_scaled[..., pair_columns]
        augmented.add_(
            (
                multiplicity[None, None, :, None]
                * quadratic_by_pair[None, :, :, None]
                * pair_query[..., None]
                * h2_query
            ).sum(dim=2)
        )
        numerator[:, :, token, :] = augmented[..., :value_dimension]
        denominator[:, :, token, :] = augmented[..., value_dimension:]
    output = (numerator / denominator).to(dtype=v.dtype)
    return output, numerator, denominator


def grouped_causal_forward_diagnostic_witness(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    constant: torch.Tensor,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    *,
    scale: float,
) -> GroupedCausalForwardDiagnosticWitness:
    """Inspect the CPU-small physical H recurrence without entering dispatch.

    This is a diagnostic oracle only.  It intentionally builds H before it
    reads any coefficient tensor, detaches every recorded value, and is never
    called by the normal forward or autograd routes.
    """
    groups, batch_size, key_value_heads, value_dimension, head_dimension = (
        _validate_inputs(q, k, v, dim_groups, constant, linear, quadratic, scale=scale)
    )
    require_cpu_diagnostic_geometry(q, k, v)
    with torch.no_grad():
        query_heads = q.shape[1]
        token_count = q.shape[2]
        pair_rows, pair_columns, multiplicity = canonical_packed_pair_metadata(
            head_dimension, device=q.device
        )
        pair_count = pair_rows.numel()
        query_to_key_value = torch.arange(query_heads, device=q.device) // (
            query_heads // key_value_heads
        )
        q_work = q.detach().float()
        k_work = k.detach().float()
        v_work = v.detach().float()
        h0 = torch.zeros(
            (batch_size, key_value_heads, value_dimension + 1),
            device=q.device,
            dtype=torch.float32,
        )
        h1 = torch.zeros(
            (batch_size, key_value_heads, head_dimension, value_dimension + 1),
            device=q.device,
            dtype=torch.float32,
        )
        h2 = torch.zeros(
            (batch_size, key_value_heads, pair_count, value_dimension + 1),
            device=q.device,
            dtype=torch.float32,
        )
        numerator = torch.empty(
            (batch_size, query_heads, token_count, value_dimension),
            device=q.device,
            dtype=torch.float32,
        )
        denominator = torch.empty(
            (batch_size, query_heads, token_count, 1),
            device=q.device,
            dtype=torch.float32,
        )
        stages: list[GroupedCausalForwardTokenWitness] = []

        # H construction has no factor/A/B/C dependency.  Coefficients are
        # read below, only after the same-token inclusive H state exists.
        for macro_start in range(0, token_count, DIAGNOSTIC_MACRO_TOKEN_BLOCK):
            macro_stop = min(macro_start + DIAGNOSTIC_MACRO_TOKEN_BLOCK, token_count)
            for token in range(macro_start, macro_stop):
                augmented_value = torch.empty(
                    (batch_size, key_value_heads, value_dimension + 1),
                    device=q.device,
                    dtype=torch.float32,
                )
                augmented_value[..., :value_dimension] = v_work[:, :, token, :]
                augmented_value[..., value_dimension] = 1.0
                key_token = k_work[:, :, token, :]
                h0_increment = augmented_value
                h1_increment = key_token.unsqueeze(-1) * augmented_value.unsqueeze(-2)
                key_pair = key_token[..., pair_rows] * key_token[..., pair_columns]
                h2_increment = key_pair.unsqueeze(-1) * augmented_value.unsqueeze(-2)
                h0.add_(h0_increment)
                h1.add_(h1_increment)
                h2.add_(h2_increment)

                constant_work = constant.detach().float()
                linear_work = linear.detach().float()
                quadratic_work = quadratic.detach().float()
                linear_by_dimension = linear_work.gather(1, groups)
                pair_row_groups = groups[:, pair_rows]
                pair_column_groups = groups[:, pair_columns]
                quadratic_by_pair = quadratic_work[
                    torch.arange(query_heads, device=q.device)[:, None],
                    pair_row_groups,
                    pair_column_groups,
                ]
                query_scaled = q_work[:, :, token, :] * float(scale)
                h0_query = h0.index_select(1, query_to_key_value)
                h1_query = h1.index_select(1, query_to_key_value)
                h2_query = h2.index_select(1, query_to_key_value)
                constant_contraction = constant_work[None, :, None] * h0_query
                linear_contraction = (
                    query_scaled[:, :, :, None]
                    * h1_query
                    * linear_by_dimension[None, :, :, None]
                ).sum(dim=2)
                pair_query = (
                    query_scaled[..., pair_rows] * query_scaled[..., pair_columns]
                )
                quadratic_contraction = (
                    multiplicity[None, None, :, None]
                    * quadratic_by_pair[None, :, :, None]
                    * pair_query[..., None]
                    * h2_query
                ).sum(dim=2)
                pre_normalized = (
                    constant_contraction + linear_contraction + quadratic_contraction
                )
                numerator[:, :, token, :] = pre_normalized[..., :value_dimension]
                denominator[:, :, token, :] = pre_normalized[..., value_dimension:]
                stages.append(
                    GroupedCausalForwardTokenWitness(
                        token_index=token,
                        macro_index=macro_start // DIAGNOSTIC_MACRO_TOKEN_BLOCK,
                        h0_increment=h0_increment.clone(),
                        h1_increment=h1_increment.clone(),
                        h2_increment=h2_increment.clone(),
                        h0_inclusive=h0.clone(),
                        h1_inclusive=h1.clone(),
                        h2_inclusive=h2.clone(),
                        constant_contraction=constant_contraction.clone(),
                        linear_contraction=linear_contraction.clone(),
                        quadratic_contraction=quadratic_contraction.clone(),
                        pre_normalized_diagonal_contraction=pre_normalized.clone(),
                    )
                )
        return GroupedCausalForwardDiagnosticWitness(
            physical_path=PHYSICAL_PATH_IDENTIFIER,
            pair_rows=pair_rows.clone(),
            pair_columns=pair_columns.clone(),
            pair_multiplicity=multiplicity.clone(),
            query_to_key_value=query_to_key_value.clone(),
            tokens=tuple(stages),
            numerator=numerator.clone(),
            denominator=denominator.clone(),
        )


__all__ = (
    "grouped_causal_forward_diagnostic_witness",
    "grouped_causal_streaming_forward",
)
