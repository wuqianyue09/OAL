"""No-token-space-N² H0/H1/H2 reference for grouped quadratic attention."""

from __future__ import annotations

import torch


def _packed_feature_pairs(
    head_dimension: int, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tril_indices(
        head_dimension,
        head_dimension,
        device=device,
    )


def _build_grouped_states(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build KV-owned FP states, using packed symmetric H2 coordinates."""
    batch_size, key_value_heads, sequence_length, head_dimension = k.shape
    value_dimension = v.shape[-1]
    pair_rows, pair_columns = _packed_feature_pairs(
        head_dimension,
        device=k.device,
    )

    augmented_value = torch.cat(
        (
            v,
            torch.ones(
                (batch_size, key_value_heads, sequence_length, 1),
                dtype=v.dtype,
                device=v.device,
            ),
        ),
        dim=-1,
    )
    h0 = augmented_value
    h1 = k.unsqueeze(-1) * augmented_value.unsqueeze(-2)
    pair_product = k[..., pair_rows] * k[..., pair_columns]
    h2 = pair_product.unsqueeze(-1) * augmented_value.unsqueeze(-2)

    if causal:
        return (
            h0.cumsum(dim=2),
            h1.cumsum(dim=2),
            h2.cumsum(dim=2),
        )
    return (
        h0.sum(dim=2),
        h1.sum(dim=2),
        h2.sum(dim=2),
    )


def _select_query_head_coefficients(
    dim_groups: torch.Tensor,
    linear_coefficients: torch.Tensor,
    quadratic_coefficients: torch.Tensor,
    *,
    head_dimension: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select B/C coefficients for every feature and packed feature pair."""
    if dim_groups.ndim == 1:
        dim_groups = dim_groups.unsqueeze(0).expand(
            linear_coefficients.shape[0],
            -1,
        )
    # The public/kernel metadata contract is canonical int32, but PyTorch's
    # gather and advanced-indexing implementations require long indices on
    # some supported versions/backends.  Keep the conversion local to the
    # reference indexing boundary; group values and validation semantics do
    # not change.
    index_groups = dim_groups.to(dtype=torch.int64)
    query_heads = dim_groups.shape[0]
    pair_rows, pair_columns = _packed_feature_pairs(
        head_dimension,
        device=dim_groups.device,
    )
    linear_by_dimension = linear_coefficients.gather(1, index_groups)
    row_groups = index_groups[:, pair_rows]
    column_groups = index_groups[:, pair_columns]
    head_offsets = torch.arange(query_heads, device=dim_groups.device)[:, None]
    quadratic_by_pair = quadratic_coefficients[
        head_offsets,
        row_groups,
        column_groups,
    ]
    multiplicity = torch.where(
        pair_rows == pair_columns,
        torch.ones((), dtype=quadratic_by_pair.dtype, device=dim_groups.device),
        torch.full((), 2.0, dtype=quadratic_by_pair.dtype, device=dim_groups.device),
    )
    return linear_by_dimension, quadratic_by_pair, multiplicity


def grouped_quadratic_unnormalized(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    constant_coefficients: torch.Tensor,
    linear_coefficients: torch.Tensor,
    quadratic_coefficients: torch.Tensor,
    *,
    scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return grouped numerator/denominator from differentiable H0/H1/H2 states."""
    _, query_heads, _, head_dimension = q.shape
    key_value_heads = k.shape[1]
    value_dimension = v.shape[-1]
    h0_kv, h1_kv, h2_kv = _build_grouped_states(k, v, causal=causal)
    query_to_key_value = torch.arange(query_heads, device=q.device) // (
        query_heads // key_value_heads
    )
    h0 = h0_kv.index_select(1, query_to_key_value)
    h1 = h1_kv.index_select(1, query_to_key_value)
    h2 = h2_kv.index_select(1, query_to_key_value)
    if not causal:
        h0 = h0.unsqueeze(2)
        h1 = h1.unsqueeze(2)
        h2 = h2.unsqueeze(2)

    linear_by_dimension, quadratic_by_pair, multiplicity = (
        _select_query_head_coefficients(
            dim_groups,
            linear_coefficients,
            quadratic_coefficients,
            head_dimension=head_dimension,
        )
    )
    pair_rows, pair_columns = _packed_feature_pairs(
        head_dimension,
        device=q.device,
    )
    q_scaled = q * scale
    q_pair = q_scaled[..., pair_rows] * q_scaled[..., pair_columns]

    h0_values = h0[..., :value_dimension]
    h1_values = h1[..., :value_dimension]
    h2_values = h2[..., :value_dimension]
    numerator = (
        constant_coefficients[None, :, None, None] * h0_values
        + (
            q_scaled.unsqueeze(-1)
            * h1_values
            * linear_by_dimension[None, :, None, :, None]
        ).sum(dim=-2)
        + (
            q_pair.unsqueeze(-1)
            * h2_values
            * quadratic_by_pair[None, :, None, :, None]
            * multiplicity[None, None, None, :, None]
        ).sum(dim=-2)
    )

    h0_count = h0[..., value_dimension : value_dimension + 1]
    h1_count = h1[..., value_dimension]
    h2_count = h2[..., value_dimension]
    denominator = (
        constant_coefficients[None, :, None, None] * h0_count
        + (q_scaled * h1_count * linear_by_dimension[None, :, None, :]).sum(
            dim=-1, keepdim=True
        )
        + (
            q_pair
            * h2_count
            * quadratic_by_pair[None, :, None, :]
            * multiplicity[None, None, None, :]
        ).sum(dim=-1, keepdim=True)
    )
    return numerator, denominator


def grouped_quadratic_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    constant_coefficients: torch.Tensor,
    linear_coefficients: torch.Tensor,
    quadratic_coefficients: torch.Tensor,
    *,
    scale: float,
    kernel_eps: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate grouped quadratic attention without materializing token pairs."""
    numerator, denominator = grouped_quadratic_unnormalized(
        q,
        k,
        v,
        dim_groups,
        constant_coefficients,
        linear_coefficients,
        quadratic_coefficients,
        scale=scale,
        causal=causal,
    )
    if not (torch.isfinite(denominator) & (denominator > 0)).all():
        raise RuntimeError(
            "Grouped Quadratic denominator must be finite and strictly positive"
        )
    output = numerator / denominator
    return output, numerator, denominator


def _state_vjp(
    k: torch.Tensor,
    v: torch.Tensor,
    d_h0: torch.Tensor,
    d_h1: torch.Tensor,
    d_h2: torch.Tensor,
    pair_rows: torch.Tensor,
    pair_columns: torch.Tensor,
    *,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, key_value_heads, sequence_length, head_dimension = k.shape
    value_dimension = v.shape[-1]
    k_work = k.float()
    v_work = v.float()
    augmented_value = torch.cat(
        (
            v_work,
            torch.ones(
                (batch_size, key_value_heads, sequence_length, 1),
                dtype=torch.float32,
                device=v.device,
            ),
        ),
        dim=-1,
    )
    if causal:
        d_h0_token = d_h0
        d_h1_token = d_h1
        d_h2_token = d_h2
    else:
        d_h0_token = d_h0.unsqueeze(2)
        d_h1_token = d_h1.unsqueeze(2)
        d_h2_token = d_h2.unsqueeze(2)
    d_u = d_h0_token
    d_u = d_u + (k_work.unsqueeze(-1) * d_h1_token).sum(dim=3)
    pair_product = k_work[..., pair_rows] * k_work[..., pair_columns]
    d_u = d_u + (pair_product.unsqueeze(-1) * d_h2_token).sum(dim=3)
    d_v = d_u[..., :value_dimension]

    d_k = torch.einsum(
        "bhndc,bhnc->bhnd",
        d_h1_token,
        augmented_value,
    )
    d_h2_pair = torch.einsum(
        "bhnpc,bhnc->bhnp",
        d_h2_token,
        augmented_value,
    )
    row_values = k_work[..., pair_rows]
    column_values = k_work[..., pair_columns]
    diagonal = pair_rows == pair_columns
    row_derivative = torch.where(
        diagonal,
        2.0 * row_values,
        column_values,
    )
    column_derivative = torch.where(
        diagonal,
        torch.zeros_like(row_values),
        row_values,
    )
    d_k = d_k.clone()
    d_k.index_add_(-1, pair_rows, d_h2_pair * row_derivative)
    d_k.index_add_(-1, pair_columns, d_h2_pair * column_derivative)
    return d_k.to(dtype=k.dtype), d_v.to(dtype=v.dtype)


def _aggregate_quadratic_pair_gradients(
    pair_gradients: torch.Tensor,
    dim_groups: torch.Tensor,
    pair_rows: torch.Tensor,
    pair_columns: torch.Tensor,
    *,
    gmax: int,
) -> torch.Tensor:
    """Reduce feature-pair partials into the selected group-pair coefficients."""
    query_heads = pair_gradients.shape[0]
    rows = pair_rows.to(dtype=torch.int64)
    columns = pair_columns.to(dtype=torch.int64)
    if dim_groups.ndim == 1:
        row_groups = dim_groups.index_select(0, rows).to(dtype=torch.int64)
        column_groups = dim_groups.index_select(0, columns).to(dtype=torch.int64)
        row_groups = row_groups.unsqueeze(0).expand(query_heads, -1)
        column_groups = column_groups.unsqueeze(0).expand(query_heads, -1)
    else:
        row_groups = dim_groups[:, rows].to(dtype=torch.int64)
        column_groups = dim_groups[:, columns].to(dtype=torch.int64)
    head_offsets = torch.arange(
        query_heads,
        device=pair_gradients.device,
        dtype=torch.int64,
    )[:, None] * (gmax * gmax)
    flat_indices = head_offsets + row_groups * gmax + column_groups
    reduced = torch.zeros(
        query_heads * gmax * gmax,
        dtype=pair_gradients.dtype,
        device=pair_gradients.device,
    ).index_add(
        0,
        flat_indices.reshape(-1),
        pair_gradients.reshape(-1),
    )
    return reduced.reshape(query_heads, gmax, gmax)
