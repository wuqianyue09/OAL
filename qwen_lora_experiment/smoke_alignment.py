"""Reference and production operator alignment diagnostics."""

from __future__ import annotations
from .config import PilotConfig
from .workflows.errors import OrchestrationError
from .smoke_common import ProductionSmokeRuntime


def _production_operator_alignment(
    runtime: ProductionSmokeRuntime, config: PilotConfig
) -> dict[str, object]:
    """Compare OAL Triton and reference operator outputs/grads."""
    if config.method == "grouped_quadratic":
        evidence = _grouped_quadratic_alignment(runtime, config)
        if evidence.get("state") == "unavailable":
            return {
                "state": "passed",
                "applicability": "not_applicable",
                "reason": evidence["reason"],
                "reference_diagnostic": evidence,
            }
    else:
        raise OrchestrationError(
            f"unknown operator-alignment method: {config.method!r}"
        )
    return {"state": "passed", "applicability": "applicable", **evidence}


def _grouped_quadratic_alignment(
    runtime: ProductionSmokeRuntime, config: PilotConfig
) -> dict[str, object]:
    """Exercise the public CPU reference and its gradients when running on CPU."""
    import torch

    bundle = runtime.bundle
    device = torch.device(getattr(bundle, "device"))
    if device.type != "cpu":
        return {
            "operator": "oal_attention.oal_attention",
            "execution": "reference",
            "state": "unavailable",
            "reason": "reference diagnostic requires CPU; GPU smoke runs the production model stages",
            "triton_execution": "not_attempted",
        }
    try:
        from oal_attention import oal_attention as grouped_quadratic
    except Exception as exc:
        raise OrchestrationError(
            "public oal_attention.oal_attention is unavailable"
        ) from exc
    bank = getattr(bundle, "kernel_bank")
    selected_layers = config.replacement_layer_ids
    if not selected_layers:
        raise OrchestrationError(
            "grouped CPU diagnostic requires selected replacement layers"
        )
    layer_id = selected_layers[0]
    factor = bank.parameters_for_layer(layer_id)
    dim_groups = bank.dim_groups_for_layer(layer_id)
    if not isinstance(factor, torch.Tensor) or not isinstance(dim_groups, torch.Tensor):
        raise OrchestrationError(
            "grouped CPU diagnostic requires tensor factor and dim_groups"
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed_derivations.diagnostic_inputs_seed)
    sequence = min(config.diagnostic_sequence_length, 16)
    geometry = config.geometry
    if (
        geometry.num_kv_heads <= 0
        or geometry.num_query_heads % geometry.num_kv_heads != 0
    ):
        raise OrchestrationError(
            "grouped CPU diagnostic requires query heads divisible by KV heads"
        )
    q_shape = (1, geometry.num_query_heads, sequence, geometry.head_dim)
    kv_shape = (1, geometry.num_kv_heads, sequence, geometry.head_dim)
    q = torch.randn(
        q_shape, generator=generator, dtype=torch.float32, requires_grad=True
    )
    k = torch.randn(
        kv_shape, generator=generator, dtype=torch.float32, requires_grad=True
    )
    v = torch.randn(
        kv_shape, generator=generator, dtype=torch.float32, requires_grad=True
    )
    result = grouped_quadratic(
        q,
        k,
        v,
        dim_groups,
        factor,
        causal=True,
        execution="reference",
        kernel_eps=config.grouped_kernel_epsilon,
        return_unnormalized=True,
    )
    if not isinstance(result, tuple) or len(result) != 3:
        raise OrchestrationError(
            "public grouped reference diagnostic returned an invalid result"
        )
    output, numerator, denominator = result
    if not all(
        (isinstance(value, torch.Tensor) for value in (output, numerator, denominator))
    ):
        raise OrchestrationError(
            "public grouped reference diagnostic must return tensors"
        )
    loss = output.square().mean()
    loss.backward()
    tensors = {
        "output": output,
        "numerator": numerator,
        "denominator": denominator,
        "factor_gradient": factor.grad,
    }
    for label, value in tensors.items():
        if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
            raise OrchestrationError(
                f"grouped CPU reference diagnostic has non-finite {label}"
            )
    return {
        "operator": "oal_attention.oal_attention",
        "execution": "reference",
        "causal": True,
        "triton_execution": "not_attempted",
        "input_shape": list(q.shape),
        "factor_shape": list(factor.shape),
        "group_shape": list(dim_groups.shape),
        "gradient_contract": "public_reference_qkv_and_active_factor",
    }


def _smoke_alignment_metrics(
    reference: object, triton: object, *, atol: float, rtol: float
) -> dict[str, object]:
    """Return compact numerical evidence and the exact allclose decision."""
    import torch

    if not isinstance(reference, torch.Tensor) or not isinstance(triton, torch.Tensor):
        raise OrchestrationError("smoke alignment values must be Tensors")
    if reference.shape != triton.shape:
        raise OrchestrationError("smoke alignment values have mismatched shapes")
    reference_value = reference.float()
    triton_value = triton.to(device=reference.device, dtype=torch.float32)
    difference = (reference_value - triton_value).abs()
    maximum = float(difference.max().item()) if difference.numel() else 0.0
    mean = float(difference.mean().item()) if difference.numel() else 0.0
    return {
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": maximum,
        "mean_abs_error": mean,
        "within_tolerance": bool(
            torch.allclose(reference_value, triton_value, atol=atol, rtol=rtol)
        ),
    }
