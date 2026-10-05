"""Contract and mathematical tests for the grouped quadratic operator."""

from __future__ import annotations
import json
import math
from pathlib import Path
import pytest
import torch
from oal_attention import oal_attention as grouped_quadratic
from oal_attention.grouped_quadratic import (
    _canonicalize_dim_groups,
    _expand_factor_coefficients,
)


def _packed_to_lower(factor: torch.Tensor, gmax: int) -> torch.Tensor:
    side = gmax + 1
    rows, columns = torch.tril_indices(side, side, device=factor.device)
    if factor.ndim == 1:
        lower = torch.zeros((side, side), dtype=factor.dtype, device=factor.device)
        return lower.index_put((rows, columns), factor)
    lower = torch.zeros(
        (factor.shape[0], side, side), dtype=factor.dtype, device=factor.device
    )
    return lower.index_put(
        (torch.arange(factor.shape[0], device=factor.device)[:, None], rows, columns),
        factor,
    )


def _dense_grouped_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dim_groups: torch.Tensor,
    factor: torch.Tensor,
    *,
    scale: float,
    kernel_eps: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent short-sequence oracle; token-pair tensors stay test-local."""
    batch, query_heads, sequence, dimension = q.shape
    key_heads = k.shape[1]
    groups = (
        dim_groups
        if dim_groups.ndim == 2
        else dim_groups.unsqueeze(0).expand(query_heads, -1)
    )
    gmax = int((factor.shape[-1] * 2) ** 0.5) - 1
    lower = _packed_to_lower(factor, gmax)
    if lower.ndim == 2:
        lower = lower.unsqueeze(0).expand(query_heads, -1, -1)
    head_map = torch.arange(query_heads, device=q.device) // (query_heads // key_heads)
    mapped_k = k.index_select(1, head_map)
    mapped_v = v.index_select(1, head_map)
    q_scaled = q.to(lower.dtype) * scale
    k_work = mapped_k.to(lower.dtype)
    value_work = mapped_v.to(lower.dtype)
    contributions = q_scaled[:, :, :, None, :] * k_work[:, :, None, :, :]
    z = torch.stack(
        [
            torch.stack(
                [
                    contributions[:, h, ..., groups[h] == group_id].sum(dim=-1)
                    for group_id in range(gmax)
                ],
                dim=-1,
            )
            for h in range(query_heads)
        ],
        dim=1,
    )
    features = torch.cat((torch.ones_like(z[..., :1]), z), dim=-1)
    projected = torch.einsum("bhijc,hcs->bhijs", features, lower)
    weights = projected.square().sum(dim=-1) + kernel_eps
    if causal:
        causal_mask = torch.ones(
            (sequence, sequence), dtype=torch.bool, device=q.device
        ).tril()
        weights = weights.masked_fill(~causal_mask, 0.0)
    numerator = torch.einsum("bhij,bhjv->bhiv", weights, value_work)
    denominator = weights.sum(dim=-1, keepdim=True)
    return (numerator / denominator, numerator, denominator)


def _case(
    *,
    batch: int = 2,
    query_heads: int = 4,
    key_heads: int = 2,
    sequence: int = 3,
    dimension: int = 4,
    value_dimension: int = 3,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20260802)
    return (
        torch.randn(
            batch, query_heads, sequence, dimension, generator=generator, dtype=dtype
        ),
        torch.randn(
            batch, key_heads, sequence, dimension, generator=generator, dtype=dtype
        ),
        torch.randn(
            batch,
            key_heads,
            sequence,
            value_dimension,
            generator=generator,
            dtype=dtype,
        ),
    )


def _two_group_mapping(*, query_heads: int = 4, dimension: int = 4) -> torch.Tensor:
    mapping = torch.ones((query_heads, dimension), dtype=torch.int32)
    mapping[:, :2] = 0
    return mapping


def _fixed_two_group_factor(
    *, dtype: torch.dtype = torch.float64, device: torch.device | str = "cpu"
) -> torch.Tensor:
    return torch.tensor((1.0, 0.5, 0.5, 0.5, 0.5, 0.0), dtype=dtype, device=device)


def _cross_group_factor(
    *, dtype: torch.dtype = torch.float64, device: torch.device | str = "cpu"
) -> torch.Tensor:
    return torch.tensor((1.1, 0.5, 0.9, 0.4, 0.8, 0.7), dtype=dtype, device=device)


@pytest.mark.parametrize("causal", [True])
def test_public_grouped_quadratic_matches_dense_oracle(causal: bool) -> None:
    q, k, v = _case()
    groups = _two_group_mapping()
    factor = _cross_group_factor()
    actual, numerator, denominator = grouped_quadratic(
        q,
        k,
        v,
        groups,
        factor,
        scale=0.37,
        kernel_eps=0.07,
        causal=causal,
        execution="reference",
        return_unnormalized=True,
    )
    expected, expected_numerator, expected_denominator = _dense_grouped_oracle(
        q, k, v, groups, factor, scale=0.37, kernel_eps=0.07, causal=causal
    )
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(numerator, expected_numerator, atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(
        denominator, expected_denominator, atol=1e-11, rtol=1e-10
    )
    assert numerator.dtype == torch.float64
    assert denominator.dtype == torch.float64
    assert torch.isfinite(denominator).all()
    assert (denominator > 0).all()


def test_fixed_initialization_is_independent_of_legal_group_partition() -> None:
    q, k, v = _case(query_heads=1, key_heads=1, dimension=4)
    groups_a = torch.tensor((0, 0, 1, 1), dtype=torch.int64)
    groups_b = torch.tensor((0, 1, 0, 1), dtype=torch.int64)
    factor = _fixed_two_group_factor()
    output_a = grouped_quadratic(q, k, v, groups_a, factor, execution="reference")
    output_b = grouped_quadratic(q, k, v, groups_b, factor, execution="reference")
    torch.testing.assert_close(output_a, output_b, atol=1e-11, rtol=1e-10)


def test_cross_group_coefficient_changes_result() -> None:
    q, k, v = _case(query_heads=1, key_heads=1, dimension=4)
    groups = torch.tensor((0, 0, 1, 1), dtype=torch.int32)
    factor = _cross_group_factor()
    no_cross_factor = factor.clone()
    no_cross_factor[3] = 0.0
    actual = grouped_quadratic(q, k, v, groups, factor, execution="reference")
    no_cross = grouped_quadratic(
        q, k, v, groups, no_cross_factor, execution="reference"
    )
    assert not torch.allclose(actual, no_cross)


@pytest.mark.parametrize(
    ("group_shape", "factor_shape"),
    [
        ("shared", "shared"),
        ("shared", "per_head"),
        ("per_head", "shared"),
        ("per_head", "per_head"),
    ],
)
def test_group_and_factor_broadcast_combinations(
    group_shape: str, factor_shape: str
) -> None:
    q, k, v = _case()
    groups = _two_group_mapping()
    factor = _cross_group_factor()
    passed_groups = groups[0] if group_shape == "shared" else groups
    passed_factor = (
        factor if factor_shape == "shared" else factor.expand(q.shape[1], -1).clone()
    )
    actual = grouped_quadratic(
        q, k, v, passed_groups, passed_factor, execution="reference"
    )
    expected, _, _ = _dense_grouped_oracle(
        q,
        k,
        v,
        passed_groups,
        passed_factor,
        scale=q.shape[-1] ** (-0.5),
        kernel_eps=1e-06,
        causal=True,
    )
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)


def test_factor_and_qkv_gradients_match_dense_oracle() -> None:
    q, k, v = _case(batch=1, sequence=3, dtype=torch.float64)
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    groups = _two_group_mapping(query_heads=q.shape[1], dimension=q.shape[-1])
    factor = _cross_group_factor().requires_grad_()
    upstream = torch.randn_like(q[..., : v.shape[-1]])
    actual = grouped_quadratic(
        q, k, v, groups, factor, scale=0.31, kernel_eps=0.09, execution="reference"
    )
    actual.backward(upstream)
    actual_gradients = tuple(
        (tensor.grad.detach().clone() for tensor in (q, k, v, factor))
    )
    q_expected, k_expected, v_expected = (
        tensor.detach().clone().requires_grad_() for tensor in (q, k, v)
    )
    factor_expected = factor.detach().clone().requires_grad_()
    expected, _, _ = _dense_grouped_oracle(
        q_expected,
        k_expected,
        v_expected,
        groups,
        factor_expected,
        scale=0.31,
        kernel_eps=0.09,
        causal=True,
    )
    expected.backward(upstream)
    expected_gradients = (
        q_expected.grad,
        k_expected.grad,
        v_expected.grad,
        factor_expected.grad,
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, atol=1e-10, rtol=1e-09
        )


def test_reference_grouped_quadratic_passes_gradcheck() -> None:
    q, k, v = _case(
        batch=1, query_heads=1, key_heads=1, sequence=2, dimension=2, value_dimension=1
    )
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    factor = torch.tensor(
        (1.0, 0.4, 0.8, 0.2, 0.3, 0.7), dtype=torch.float64, requires_grad=True
    )
    groups = torch.tensor((0, 1), dtype=torch.int32)

    def function(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        live_factor: torch.Tensor,
    ) -> torch.Tensor:
        return grouped_quadratic(
            query,
            key,
            value,
            groups,
            live_factor,
            kernel_eps=0.1,
            execution="reference",
        )

    assert torch.autograd.gradcheck(
        function, (q, k, v, factor), atol=1e-06, rtol=0.0001
    )


def test_broadcast_factor_gradient_reduces_across_query_heads() -> None:
    q, k, v = _case(batch=1)
    groups = _two_group_mapping()
    shared_factor = _cross_group_factor().requires_grad_()
    output = grouped_quadratic(q, k, v, groups, shared_factor, execution="reference")
    output.square().sum().backward()
    shared_gradient = shared_factor.grad.detach().clone()
    per_head_factor = (
        _cross_group_factor().expand(q.shape[1], -1).clone().requires_grad_()
    )
    per_head_output = grouped_quadratic(
        q, k, v, groups, per_head_factor, execution="reference"
    )
    per_head_output.square().sum().backward()
    torch.testing.assert_close(
        shared_gradient, per_head_factor.grad.sum(dim=0), atol=1e-10, rtol=1e-09
    )


def test_inactive_padded_factor_entries_do_not_change_per_head_output_or_gradient() -> (
    None
):
    q, k, v = _case(query_heads=2, key_heads=1, dimension=4)
    groups = torch.tensor(((0, 0, 1, 1), (0, 0, 0, 0)), dtype=torch.int32)
    factor = _cross_group_factor().expand(2, -1).clone().requires_grad_()
    output = grouped_quadratic(q, k, v, groups, factor, execution="reference")
    loss = output[:, 1].square().sum()
    loss.backward()
    assert torch.equal(factor.grad[0], torch.zeros_like(factor.grad[0]))
    torch.testing.assert_close(factor.grad[1, 3:], torch.zeros_like(factor.grad[1, 3:]))
    changed = factor.detach().clone()
    changed[1, 3:] += 100.0
    changed_output = grouped_quadratic(q, k, v, groups, changed, execution="reference")
    torch.testing.assert_close(
        output[:, 1], changed_output[:, 1], atol=1e-11, rtol=1e-10
    )


@pytest.mark.parametrize(
    "bad_groups",
    [
        torch.tensor((0.0, 0.0, 1.0, 1.0)),
        torch.tensor((-1, 0, 1, 1), dtype=torch.int64),
        torch.tensor((0, 2, 2, 2), dtype=torch.int64),
        torch.tensor((1, 1, 2, 2), dtype=torch.int64),
        torch.tensor((0, 0, 2, 2), dtype=torch.int64),
        torch.tensor((0, 0, 2**32, 2**32), dtype=torch.int64),
    ],
)
def test_invalid_dim_groups_are_rejected(bad_groups: torch.Tensor) -> None:
    q, k, v = _case(query_heads=1, key_heads=1)
    with pytest.raises((TypeError, ValueError), match="dim_groups|group"):
        grouped_quadratic(
            q, k, v, bad_groups, _cross_group_factor(), execution="reference"
        )


def test_invalid_shapes_and_factor_lengths_are_rejected() -> None:
    q, k, v = _case()
    groups = _two_group_mapping()
    with pytest.raises(ValueError, match="query-head"):
        grouped_quadratic(
            q, k, v, groups[:2], _cross_group_factor(), execution="reference"
        )
    with pytest.raises(ValueError, match="triangular|factor"):
        grouped_quadratic(q, k, v, groups, torch.ones(5), execution="reference")
    with pytest.raises(ValueError, match="factor"):
        grouped_quadratic(q, k, v, groups, torch.ones(2, 6), execution="reference")
    with pytest.raises(TypeError, match="floating"):
        grouped_quadratic(
            q, k, v, groups, torch.ones(6, dtype=torch.int64), execution="reference"
        )


@pytest.mark.parametrize("kernel_eps", [0.0, -1.0, math.inf, 1e-50])
def test_invalid_kernel_epsilon_is_rejected(kernel_eps: float) -> None:
    q, k, v = _case(query_heads=1, key_heads=1, dtype=torch.float32)
    with pytest.raises((TypeError, ValueError), match="kernel_eps"):
        grouped_quadratic(
            q,
            k,
            v,
            torch.tensor((0, 0, 1, 1), dtype=torch.int32),
            _cross_group_factor(dtype=torch.float32),
            kernel_eps=kernel_eps,
            execution="reference",
        )


def test_invalid_execution_and_triton_cpu_dispatch_are_rejected() -> None:
    q, k, v = _case(query_heads=1, key_heads=1)
    groups = torch.tensor((0, 0, 1, 1), dtype=torch.int32)
    factor = _cross_group_factor()
    with pytest.raises(ValueError, match="execution"):
        grouped_quadratic(q, k, v, groups, factor, execution="torch")
    with pytest.raises(RuntimeError, match="CUDA|Triton"):
        grouped_quadratic(q, k, v, groups, factor, execution="triton")


def test_output_dtype_only_changes_normalized_output() -> None:
    q, k, v = _case(query_heads=1, key_heads=1, dtype=torch.float32)
    groups = torch.tensor((0, 0, 1, 1), dtype=torch.int32)
    output, numerator, denominator = grouped_quadratic(
        q,
        k,
        v,
        groups,
        _cross_group_factor(dtype=torch.float32),
        execution="reference",
        output_dtype=torch.float16,
        return_unnormalized=True,
    )
    assert output.dtype == torch.float16
    assert numerator.dtype == torch.float32
    assert denominator.dtype == torch.float32


def test_default_scale_is_d_inverse_sqrt_and_reference_supports_d_over_64() -> None:
    q, k, v = _case(
        batch=1, query_heads=1, key_heads=1, sequence=2, dimension=65, value_dimension=1
    )
    groups = torch.zeros(65, dtype=torch.int32)
    factor = torch.tensor((1.0, 0.5, 0.5), dtype=torch.float64)
    default_output = grouped_quadratic(q, k, v, groups, factor, execution="reference")
    explicit_output = grouped_quadratic(
        q, k, v, groups, factor, scale=65 ** (-0.5), execution="reference"
    )
    torch.testing.assert_close(default_output, explicit_output, atol=1e-11, rtol=1e-10)


def test_default_execution_is_reference_on_non_cuda_devices() -> None:
    q, k, v = _case(query_heads=1, key_heads=1)
    groups = torch.tensor((0, 0, 1, 1), dtype=torch.int32)
    factor = _cross_group_factor()
    default_output = grouped_quadratic(q, k, v, groups, factor)
    reference_output = grouped_quadratic(q, k, v, groups, factor, execution="reference")
    torch.testing.assert_close(default_output, reference_output)


def test_reference_uses_int64_group_indices_for_torch_indexing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q, k, v = _case(query_heads=1, key_heads=1)
    groups = torch.tensor((0, 0, 1, 1), dtype=torch.int32)
    factor = _cross_group_factor()
    original_gather = torch.Tensor.gather
    observed_index_dtypes: list[torch.dtype] = []

    def gather_with_dtype_check(
        tensor: torch.Tensor,
        dimension: int,
        index: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        observed_index_dtypes.append(index.dtype)
        return original_gather(tensor, dimension, index, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "gather", gather_with_dtype_check)
    grouped_quadratic(q, k, v, groups, factor, execution="reference")
    assert observed_index_dtypes
    assert all((dtype == torch.int64 for dtype in observed_index_dtypes))


def test_shared_dim_groups_remain_a_logical_broadcast() -> None:
    canonical = _canonicalize_dim_groups(
        torch.tensor((0, 0, 1, 1), dtype=torch.int64),
        query_heads=4,
        head_dimension=4,
        gmax=2,
        q_device=torch.device("cpu"),
    )
    assert canonical.ndim == 1
    assert canonical.shape == (4,)


def test_packed_factor_expansion_matches_psd_coefficients() -> None:
    factor = _cross_group_factor()
    constant, linear, quadratic = _expand_factor_coefficients(
        factor, query_heads=1, gmax=2, kernel_eps=0.07, working_dtype=torch.float64
    )
    lower = torch.tensor(
        ((1.1, 0.0, 0.0), (0.5, 0.9, 0.0), (0.4, 0.8, 0.7)), dtype=torch.float64
    )
    matrix = lower @ lower.transpose(-1, -2)
    torch.testing.assert_close(constant, matrix[0, 0].view(1) + 0.07)
    torch.testing.assert_close(linear, 2 * matrix[0, 1:].view(1, 2))
    torch.testing.assert_close(quadratic, matrix[1:, 1:].view(1, 2, 2))
    assert quadratic.is_contiguous()


def test_grouped_production_sources_keep_token_complexity_linear() -> None:
    package_root = Path(__file__).parents[2] / "oal_attention"
    grouped_sources = (
        package_root / "grouped_quadratic.py",
        package_root / "grouped_reference.py",
        package_root / "triton" / "grouped_quadratic.py",
        package_root / "triton" / "grouped_quadratic_backward.py",
    )
    forbidden = ("QK^T", "repeat_interleave", "[N,N]", "[N,N,G]")
    for source_path in grouped_sources:
        source = source_path.read_text()
        assert not any((token in source for token in forbidden)), source_path


def test_packed_h2_pair_count_is_triangular() -> None:
    for dimension in (1, 2, 16, 17, 63, 64, 65):
        assert dimension * (dimension + 1) // 2 == len(
            torch.tril_indices(dimension, dimension)[0]
        )
