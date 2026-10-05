"""Fail-closed public Adaptive MultiGroup Quadratic attention bridge.

The experiment calls only ``oal_attention.oal_attention``.  Training
and NLL always request its planner-supported Triton execution.  A private,
temporary quality-scoring context may explicitly select the public reference
API for bounded variable-length endpoints; it is never a Triton fallback.
"""

from __future__ import annotations
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
import importlib
from typing import TYPE_CHECKING, Literal, cast
import torch
from torch import Tensor, nn
from ..config import GROUPED_BASE_METHODS
from ..kernel_parameters import GROUPED_FACTOR_SIZE, HEAD_DIM, KernelParameterBank
from .common import (
    AttentionContractError,
    QwenAttentionAdapter,
    is_checkpoint_early_stop_exception,
)

if TYPE_CHECKING:
    from .hd_runtime import HDAttentionRuntime


class GroupedQuadraticAttentionError(AttentionContractError):
    """Raised when the public grouped runtime rejects the experiment call."""


_GROUPED_FACTOR_REFIT_SESSION: ContextVar[bool] = ContextVar(
    "grouped_factor_refit_session", default=False
)


@contextmanager
def grouped_quadratic_factor_refit_session() -> Iterator[None]:
    """Allow frozen factor leaves while a targeted refit propagates through them.

    The normal training contract still requires every factor packing to carry
    gradients.  Targeted refit is the narrow exception: non-target layers are
    deliberately frozen but must execute so later selected layers receive a
    valid end-to-end VJP.
    """
    token = _GROUPED_FACTOR_REFIT_SESSION.set(True)
    try:
        yield
    finally:
        _GROUPED_FACTOR_REFIT_SESSION.reset(token)


def grouped_quadratic_triton_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    parameter_bank: KernelParameterBank,
    layer_id: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Invoke the public Triton API with one packed factor.

    Unsupported device, dtype, geometry, or plan errors are surfaced with
    context. No execution mode other than ``"triton"`` is requested here.
    """
    checked_q, checked_k, checked_v = _validate_qkv(q, k, v, layer_id=layer_id)
    factor, dim_groups, epsilon = _grouped_layer_state(
        parameter_bank, layer_id=layer_id, device=checked_q.device
    )
    checked_q, factor = _prepare_factor_refit_operator_inputs(
        checked_q, checked_k, checked_v, factor
    )
    operator = _resolve_grouped_public_callable(layer_id=layer_id)
    return _invoke_grouped_callable(
        operator,
        checked_q,
        checked_k,
        checked_v,
        dim_groups,
        factor,
        execution="triton",
        epsilon=epsilon,
        layer_id=layer_id,
    )


def grouped_quadratic_hd_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    parameter_bank: KernelParameterBank,
    layer_id: int,
    hd_runtime: HDAttentionRuntime,
) -> tuple[Tensor, Tensor, Tensor]:
    """Invoke the HD family with the current layer's live grouped factor."""
    checked_q, checked_k, checked_v = _validate_qkv(q, k, v, layer_id=layer_id)
    factor, dim_groups, epsilon = _grouped_layer_state(
        parameter_bank, layer_id=layer_id, device=checked_q.device
    )
    checked_q, factor = _prepare_factor_refit_operator_inputs(
        checked_q, checked_k, checked_v, factor
    )
    try:
        result = hd_runtime.grouped(
            checked_q,
            checked_k,
            checked_v,
            dim_groups,
            factor,
            cache_key=layer_id,
            scale=HEAD_DIM ** (-0.5),
            kernel_eps=epsilon,
        )
    except Exception as exc:
        if is_checkpoint_early_stop_exception(exc):
            raise
        raise GroupedQuadraticAttentionError(
            _context(layer_id) + ": HD grouped attention call failed"
        ) from exc
    return _validate_public_result(result, checked_q, layer_id=layer_id)


def _grouped_quadratic_reference_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    parameter_bank: KernelParameterBank,
    layer_id: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Invoke the public reference only from the temporary PIQA quality context."""
    checked_q, checked_k, checked_v = _validate_qkv(q, k, v, layer_id=layer_id)
    factor, dim_groups, epsilon = _grouped_layer_state(
        parameter_bank, layer_id=layer_id, device=checked_q.device
    )
    checked_q, factor = _prepare_factor_refit_operator_inputs(
        checked_q, checked_k, checked_v, factor
    )
    operator = _resolve_grouped_public_callable(layer_id=layer_id)
    return _invoke_grouped_callable(
        operator,
        checked_q,
        checked_k,
        checked_v,
        dim_groups,
        factor,
        execution="reference",
        epsilon=epsilon,
        layer_id=layer_id,
    )


def run_grouped_quadratic_diagnostics(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    parameter_bank: KernelParameterBank,
    layer_id: int,
    hd_runtime: HDAttentionRuntime | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run the same public call under ``no_grad`` for denominator auditing."""
    with torch.no_grad():
        if hd_runtime is None:
            output, numerator, denominator = grouped_quadratic_triton_attention(
                q, k, v, parameter_bank=parameter_bank, layer_id=layer_id
            )
        else:
            output, numerator, denominator = grouped_quadratic_hd_attention(
                q,
                k,
                v,
                parameter_bank=parameter_bank,
                layer_id=layer_id,
                hd_runtime=hd_runtime,
            )
        return (output.detach(), numerator.detach(), denominator.detach())


class GroupedQuadraticQwenAttentionAdapter(QwenAttentionAdapter):
    """Qwen projection/RoPE shell backed only by the public grouped API."""

    def __init__(
        self,
        *args: object,
        grouped_parameter_bank: KernelParameterBank,
        hd_runtime: HDAttentionRuntime | None = None,
        **kwargs: object,
    ) -> None:
        _validate_grouped_bank(grouped_parameter_bank)
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_grouped_parameter_bank", grouped_parameter_bank)
        object.__setattr__(self, "_grouped_execution", "triton")
        object.__setattr__(self, "_hd_runtime", hd_runtime)

    def grouped_factor_for_layer(self) -> Tensor:
        """Return this layer's live public ``[14, 45]`` FP32 factor."""
        factor, _, _ = _grouped_layer_state(
            self._grouped_parameter_bank, layer_id=self.layer_id, device=None
        )
        return factor

    def dim_groups_for_layer(self) -> Tensor:
        """Return this layer's true-prefix int32 ``[14, 64]`` labels."""
        _, groups, _ = _grouped_layer_state(
            self._grouped_parameter_bank, layer_id=self.layer_id, device=None
        )
        return groups

    def diagnostics(
        self, q: Tensor, k: Tensor, v: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run the public operator's fixed-input no-grad diagnostic."""
        return run_grouped_quadratic_diagnostics(
            q,
            k,
            v,
            parameter_bank=self._grouped_parameter_bank,
            layer_id=self.layer_id,
            hd_runtime=self._hd_runtime,
        )

    def head_attention(self, q: Tensor, k: Tensor, v: Tensor, layer_id: int) -> Tensor:
        """Return output from the one permissible Grouped runtime call."""
        if layer_id != self.layer_id:
            raise GroupedQuadraticAttentionError(
                _context(layer_id) + ": adapter received a mismatched layer_id"
            )
        execution = self._grouped_execution
        if execution == "triton":
            if self._hd_runtime is None:
                output, _, _ = grouped_quadratic_triton_attention(
                    q,
                    k,
                    v,
                    parameter_bank=self._grouped_parameter_bank,
                    layer_id=layer_id,
                )
            else:
                output, _, _ = grouped_quadratic_hd_attention(
                    q,
                    k,
                    v,
                    parameter_bank=self._grouped_parameter_bank,
                    layer_id=layer_id,
                    hd_runtime=self._hd_runtime,
                )
        elif execution == "reference":
            output, _, _ = _grouped_quadratic_reference_attention(
                q, k, v, parameter_bank=self._grouped_parameter_bank, layer_id=layer_id
            )
        else:
            raise GroupedQuadraticAttentionError(
                _context(layer_id) + ": adapter has an invalid grouped execution mode"
            )
        return output


@contextmanager
def grouped_quadratic_reference_session(
    model: nn.Module, *, expected_layer_ids: Sequence[int]
) -> Iterator[None]:
    """Temporarily select public reference execution for one quality endpoint.

    This deliberately mutates only the already-constructed, formally admitted
    Grouped adapters.  It rejects a partial or pre-existing reference switch
    and always restores Triton after the scorer returns or raises.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("grouped reference session requires an nn.Module model")
    if isinstance(expected_layer_ids, (str, bytes)):
        raise TypeError("grouped reference layer IDs must be a sequence of integers")
    layer_ids = tuple(expected_layer_ids)
    if (
        not layer_ids
        or any((type(layer_id) is not int or layer_id < 0 for layer_id in layer_ids))
        or layer_ids != tuple(sorted(set(layer_ids)))
    ):
        raise ValueError(
            "grouped reference layer IDs must be sorted unique non-negative integers"
        )
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise GroupedQuadraticAttentionError(
            "grouped reference session cannot inspect Qwen layers"
        )
    actual_layer_ids = tuple(
        (
            layer_id
            for (layer_id, layer) in enumerate(layers)
            if isinstance(
                getattr(layer, "self_attn", None), GroupedQuadraticQwenAttentionAdapter
            )
        )
    )
    if actual_layer_ids != layer_ids:
        raise GroupedQuadraticAttentionError(
            "grouped reference session adapters do not match the frozen replacement layers"
        )
    adapters = tuple((layers[layer_id].self_attn for layer_id in layer_ids))
    if any((adapter._grouped_execution != "triton" for adapter in adapters)):
        raise GroupedQuadraticAttentionError(
            "grouped reference session requires adapters to begin in Triton mode"
        )
    for adapter in adapters:
        object.__setattr__(adapter, "_grouped_execution", "reference")
    try:
        yield
    finally:
        for adapter in adapters:
            object.__setattr__(adapter, "_grouped_execution", "triton")


@contextmanager
def grouped_quadratic_reference_quality_session(
    model: nn.Module, *, expected_layer_ids: Sequence[int]
) -> Iterator[None]:
    """PIQA compatibility alias for :func:`grouped_quadratic_reference_session`."""
    with grouped_quadratic_reference_session(
        model, expected_layer_ids=expected_layer_ids
    ):
        yield


def grouped_quadratic_reference_alignment(
    *,
    parameter_bank: KernelParameterBank,
    layer_id: int,
    device: torch.device | str,
    seed: int,
) -> dict[str, object]:
    """Check a reference formula against the N=128 Triton call.

    This is a no-grad implementation guard, not a benchmark.  On a production
    CUDA host, the explicit Triton call must satisfy the live planner before a
    variable-length reference PIQA sweep is allowed. Reviewed status is logged
    separately as evidence.
    """
    if type(seed) is not int:
        raise TypeError("grouped reference alignment seed must be an integer")
    target_device = torch.device(device)
    sequence_length = 128
    generator = torch.Generator(device=target_device)
    generator.manual_seed(seed)
    q = (
        torch.randn(
            (
                1,
                parameter_bank.geometry.num_query_heads,
                sequence_length,
                parameter_bank.geometry.head_dim,
            ),
            device=target_device,
            dtype=torch.float32,
            generator=generator,
        )
        .to(dtype=torch.bfloat16)
        .contiguous()
    )
    k = (
        torch.randn(
            (
                1,
                parameter_bank.geometry.num_kv_heads,
                sequence_length,
                parameter_bank.geometry.head_dim,
            ),
            device=target_device,
            dtype=torch.float32,
            generator=generator,
        )
        .to(dtype=torch.bfloat16)
        .contiguous()
    )
    v = (
        torch.randn(
            (
                1,
                parameter_bank.geometry.num_kv_heads,
                sequence_length,
                parameter_bank.geometry.head_dim,
            ),
            device=target_device,
            dtype=torch.float32,
            generator=generator,
        )
        .to(dtype=torch.bfloat16)
        .contiguous()
    )
    with torch.inference_mode():
        reference, _, _ = _grouped_quadratic_reference_attention(
            q, k, v, parameter_bank=parameter_bank, layer_id=layer_id
        )
        triton, _, _ = grouped_quadratic_triton_attention(
            q, k, v, parameter_bank=parameter_bank, layer_id=layer_id
        )
    if not bool(torch.isfinite(reference).all()) or not bool(
        torch.isfinite(triton).all()
    ):
        raise GroupedQuadraticAttentionError(
            _context(layer_id)
            + ": grouped reference alignment produced non-finite output"
        )
    difference = (reference.float() - triton.float()).abs()
    maximum = float(difference.max().item()) if difference.numel() else 0.0
    mean = float(difference.mean().item()) if difference.numel() else 0.0
    atol = 0.03
    rtol = 0.03
    within_tolerance = bool(
        torch.allclose(reference.float(), triton.float(), atol=atol, rtol=rtol)
    )
    if not within_tolerance:
        raise GroupedQuadraticAttentionError(
            _context(layer_id)
            + ": grouped N=128 reference-to-Triton alignment exceeded tolerance"
        )
    return {
        "schema": "grouped_reference_alignment_v1",
        "sequence_length": sequence_length,
        "layer_id": layer_id,
        "seed": seed,
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": maximum,
        "mean_abs_error": mean,
        "within_tolerance": within_tolerance,
    }


def grouped_quadratic_reference_quality_alignment(
    *,
    parameter_bank: KernelParameterBank,
    layer_id: int,
    device: torch.device | str,
    seed: int,
) -> dict[str, object]:
    """PIQA compatibility alias retaining the historical alignment schema."""
    evidence = grouped_quadratic_reference_alignment(
        parameter_bank=parameter_bank, layer_id=layer_id, device=device, seed=seed
    )
    evidence["schema"] = "grouped_piqa_reference_alignment_v1"
    return evidence


def _resolve_grouped_public_callable(*, layer_id: int) -> Callable[..., object]:
    """Resolve the OAL callable exported by the operator package."""
    try:
        package = importlib.import_module("oal_attention")
        operator = getattr(package, "oal_attention")
    except Exception as exc:
        raise GroupedQuadraticAttentionError(
            _context(layer_id) + ": could not import public oal_attention.oal_attention"
        ) from exc
    if not callable(operator):
        raise GroupedQuadraticAttentionError(
            _context(layer_id) + ": public oal_attention.oal_attention is not callable"
        )
    return cast(Callable[..., object], operator)


def _invoke_grouped_callable(
    operator: Callable[..., object],
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dim_groups: Tensor,
    factor: Tensor,
    *,
    execution: Literal["triton", "reference"],
    epsilon: float,
    layer_id: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Invoke one already-resolved callable under the fixed public contract."""
    try:
        result = operator(
            q,
            k,
            v,
            dim_groups,
            factor,
            causal=True,
            execution=execution,
            kernel_eps=epsilon,
            return_unnormalized=True,
        )
    except Exception as exc:
        if is_checkpoint_early_stop_exception(exc):
            raise
        mode = "Triton planner/call" if execution == "triton" else "reference call"
        raise GroupedQuadraticAttentionError(
            _context(layer_id) + f": public grouped_quadratic {mode} failed"
        ) from exc
    return _validate_public_result(result, q, layer_id=layer_id)


def _run_nonformal_grouped_quadratic_callable_test_helper(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    parameter_bank: KernelParameterBank,
    layer_id: int,
    grouped_test_callable: Callable[..., object],
) -> tuple[Tensor, Tensor, Tensor]:
    """Internal CPU/reference test helper; it is never used by production.

    The deliberately explicit name prevents an injected callable from being
    mistaken for the public OAL runtime path.  Orchestration rejects all
    model/runtime injection seams unless its explicit nonformal test marker is
    set, so output from this helper cannot establish formal evidence.
    """
    if not callable(grouped_test_callable):
        raise GroupedQuadraticAttentionError(
            "nonformal grouped test callable must be callable"
        )
    checked_q, checked_k, checked_v = _validate_qkv(q, k, v, layer_id=layer_id)
    factor, dim_groups, epsilon = _grouped_layer_state(
        parameter_bank, layer_id=layer_id, device=checked_q.device
    )
    checked_q, factor = _prepare_factor_refit_operator_inputs(
        checked_q, checked_k, checked_v, factor
    )
    return _invoke_grouped_callable(
        grouped_test_callable,
        checked_q,
        checked_k,
        checked_v,
        dim_groups,
        factor,
        execution="triton",
        epsilon=epsilon,
        layer_id=layer_id,
    )


def _run_nonformal_grouped_quadratic_diagnostics_test_helper(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    parameter_bank: KernelParameterBank,
    layer_id: int,
    grouped_test_callable: Callable[..., object],
) -> tuple[Tensor, Tensor, Tensor]:
    """Internal no-grad companion to the nonformal callable test helper."""
    with torch.no_grad():
        output, numerator, denominator = (
            _run_nonformal_grouped_quadratic_callable_test_helper(
                q,
                k,
                v,
                parameter_bank=parameter_bank,
                layer_id=layer_id,
                grouped_test_callable=grouped_test_callable,
            )
        )
    return (output.detach(), numerator.detach(), denominator.detach())


class _NonformalGroupedQuadraticQwenAttentionTestAdapter(
    GroupedQuadraticQwenAttentionAdapter
):
    """Test-only adapter that cannot be constructed by production assembly."""

    def __init__(
        self,
        *args: object,
        grouped_test_callable: Callable[..., object],
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not callable(grouped_test_callable):
            raise GroupedQuadraticAttentionError(
                "nonformal grouped test callable must be callable"
            )
        object.__setattr__(self, "_grouped_test_callable", grouped_test_callable)

    def diagnostics(
        self, q: Tensor, k: Tensor, v: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        return _run_nonformal_grouped_quadratic_diagnostics_test_helper(
            q,
            k,
            v,
            parameter_bank=self._grouped_parameter_bank,
            layer_id=self.layer_id,
            grouped_test_callable=self._grouped_test_callable,
        )

    def head_attention(self, q: Tensor, k: Tensor, v: Tensor, layer_id: int) -> Tensor:
        if layer_id != self.layer_id:
            raise GroupedQuadraticAttentionError(
                _context(layer_id) + ": adapter received a mismatched layer_id"
            )
        output, _, _ = _run_nonformal_grouped_quadratic_callable_test_helper(
            q,
            k,
            v,
            parameter_bank=self._grouped_parameter_bank,
            layer_id=layer_id,
            grouped_test_callable=self._grouped_test_callable,
        )
        return output


def _build_nonformal_grouped_quadratic_test_adapter(
    *args: object, grouped_test_callable: Callable[..., object], **kwargs: object
) -> GroupedQuadraticQwenAttentionAdapter:
    """Build the clearly marked nonformal adapter used by CPU unit tests."""
    return _NonformalGroupedQuadraticQwenAttentionTestAdapter(
        *args, grouped_test_callable=grouped_test_callable, **kwargs
    )


def _validate_grouped_bank(bank: object) -> KernelParameterBank:
    if not isinstance(bank, KernelParameterBank):
        raise GroupedQuadraticAttentionError(
            "grouped parameter_bank must be KernelParameterBank"
        )
    if bank.method != "grouped_quadratic":
        raise GroupedQuadraticAttentionError(
            "grouped parameter_bank.method must be 'grouped_quadratic'"
        )
    return _validate_grouped_base_bank(bank)


def _validate_grouped_base_bank(bank: object) -> KernelParameterBank:
    """Admit immutable state shared by every OAL-backed method."""
    if not isinstance(bank, KernelParameterBank):
        raise GroupedQuadraticAttentionError(
            "grouped parameter_bank must be KernelParameterBank"
        )
    if bank.method not in GROUPED_BASE_METHODS:
        raise GroupedQuadraticAttentionError(
            "grouped parameter_bank.method must be a OAL-backed method"
        )
    try:
        bank.validate_grouped_static_contract()
    except (RuntimeError, TypeError, ValueError) as exc:
        raise GroupedQuadraticAttentionError(str(exc)) from exc
    return bank


def _grouped_layer_state(
    bank: KernelParameterBank, *, layer_id: int, device: torch.device | None
) -> tuple[Tensor, Tensor, float]:
    _validate_layer_id(layer_id)
    if bank.method not in GROUPED_BASE_METHODS:
        raise GroupedQuadraticAttentionError(
            "grouped parameter_bank.method must be a OAL-backed method"
        )
    if layer_id not in bank.trainable_kernel_layer_ids:
        raise GroupedQuadraticAttentionError(
            _context(layer_id) + ": layer has no selected Adaptive MultiGroup factor"
        )
    factor = bank.parameters_for_layer(layer_id)
    groups = bank.dim_groups_for_layer(layer_id)
    if (
        tuple(factor.shape) != (bank.geometry.num_query_heads, GROUPED_FACTOR_SIZE)
        or factor.dtype is not torch.float32
        or (
            torch.is_grad_enabled()
            and (not factor.requires_grad)
            and (not _GROUPED_FACTOR_REFIT_SESSION.get())
        )
        or (not factor.is_contiguous())
    ):
        raise GroupedQuadraticAttentionError(
            _context(layer_id) + ": grouped factor must be a live FP32 [14, 45] packing"
        )
    if (
        tuple(groups.shape) != (bank.geometry.num_query_heads, bank.geometry.head_dim)
        or groups.dtype is not torch.int32
        or (not groups.is_contiguous())
    ):
        raise GroupedQuadraticAttentionError(
            _context(layer_id) + ": dim_groups must be contiguous int32 [14, 64]"
        )
    if device is not None and (factor.device != device or groups.device != device):
        raise GroupedQuadraticAttentionError(
            _context(layer_id)
            + ": grouped factor and dim_groups must share q/k/v device"
        )
    return (factor, groups, bank.grouped_kernel_epsilon_for_runtime())


def _prepare_factor_refit_operator_inputs(
    q: Tensor, k: Tensor, v: Tensor, factor: Tensor
) -> tuple[Tensor, Tensor]:
    """Select one reviewed sparse-VJP shape without changing frozen factor leaves.

    Before the first selected factor, every frozen Transformer projection has
    `requires_grad=False`.  The Triton registry admits Q-only + coefficient
    VJP, not an all-false Q/K/V/factor call under a gradient-enabled outer
    refit.  A detached factor shadow receives the operator's disposable
    coefficient gradients while the true frozen leaves remain untouched.
    """
    if not _GROUPED_FACTOR_REFIT_SESSION.get():
        return (q, factor)
    if not factor.requires_grad:
        factor = factor.detach().requires_grad_(True)
    if not (q.requires_grad or k.requires_grad or v.requires_grad):
        q = q.detach().requires_grad_(True)
    return (q, factor)


def _validate_qkv(
    q: object, k: object, v: object, *, layer_id: int
) -> tuple[Tensor, Tensor, Tensor]:
    context = _context(layer_id)
    if (
        not isinstance(q, Tensor)
        or not isinstance(k, Tensor)
        or (not isinstance(v, Tensor))
    ):
        raise GroupedQuadraticAttentionError(
            context + ": q, k, and v must be torch.Tensor values"
        )
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise GroupedQuadraticAttentionError(
            context + ": q, k, and v must use rank-4 head-first layout"
        )
    if (
        q.shape[0] <= 0
        or q.shape[2] <= 0
        or q.shape[1] <= 0
        or (q.shape[-1] != HEAD_DIM)
        or (k.shape[1] <= 0)
        or q.shape[1] % k.shape[1]
        or (tuple(k.shape[2:]) != (q.shape[2], HEAD_DIM))
        or (tuple(v.shape) != tuple(k.shape))
    ):
        raise GroupedQuadraticAttentionError(
            context
            + ": expected q [B, Hq, N, 64] and k/v [B, Hkv, N, 64] with Hq divisible by Hkv"
        )
    if not q.is_contiguous() or not k.is_contiguous() or (not v.is_contiguous()):
        raise GroupedQuadraticAttentionError(
            context + ": q, k, and v must be contiguous"
        )
    if q.device.type == "meta" or k.device != q.device or v.device != q.device:
        raise GroupedQuadraticAttentionError(
            context + ": q, k, and v must share one non-meta device"
        )
    if not q.is_floating_point() or k.dtype != q.dtype or v.dtype != q.dtype:
        raise GroupedQuadraticAttentionError(
            context + ": q, k, and v must share one floating dtype"
        )
    return (q, k, v)


def _validate_public_result(
    result: object, q: Tensor, *, layer_id: int
) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(result, tuple) or len(result) != 3:
        raise GroupedQuadraticAttentionError(
            _context(layer_id)
            + ": public grouped_quadratic must return (output, numerator, denominator)"
        )
    output, numerator, denominator = result
    if not isinstance(output, Tensor) or tuple(output.shape) != tuple(q.shape):
        raise GroupedQuadraticAttentionError(
            _context(layer_id) + ": grouped output must match q shape"
        )
    if output.dtype != q.dtype or output.device != q.device:
        raise GroupedQuadraticAttentionError(
            _context(layer_id) + ": grouped output must match q dtype and device"
        )
    if not isinstance(numerator, Tensor) or tuple(numerator.shape) != tuple(q.shape):
        raise GroupedQuadraticAttentionError(
            _context(layer_id) + ": grouped numerator must match q shape"
        )
    if not isinstance(denominator, Tensor) or tuple(denominator.shape) != (
        *q.shape[:-1],
        1,
    ):
        raise GroupedQuadraticAttentionError(
            _context(layer_id) + ": grouped denominator must have shape [B, 14, N, 1]"
        )
    for name, tensor in (
        ("output", output),
        ("numerator", numerator),
        ("denominator", denominator),
    ):
        if tensor.device != q.device or not tensor.is_floating_point():
            raise GroupedQuadraticAttentionError(
                _context(layer_id) + f": grouped {name} must be floating on q device"
            )
    return (output, numerator, denominator)


def _validate_layer_id(layer_id: int) -> None:
    if type(layer_id) is not int or layer_id < 0:
        raise GroupedQuadraticAttentionError(
            "grouped layer_id must be a non-negative integer"
        )


def _context(layer_id: int) -> str:
    return f"grouped quadratic operator layer={layer_id}"
