"""Narrow precision-aware contraction seam for private HD Block-GEMM paths."""

from __future__ import annotations

import torch

from .hd_cublas_compat import (
    HdContractionBackendIdentity,
    LoadedHdContractionBackendToken,
    _validate_loaded_token_for_contraction,
)
from .hd_block_gemm_profiling import (
    _record_hd_bmm_stage,
    _record_hd_contraction,
    _record_hd_stage,
)

_SUPPORTED_PRECISIONS = frozenset(("fp32_ieee", "bf16_tensorcore"))


def _require_precision(precision: str) -> None:
    if precision not in _SUPPORTED_PRECISIONS:
        raise ValueError("unsupported HD Block-GEMM contraction precision")


def _bmm_fp32(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    out: torch.Tensor,
    precision: str,
    backend_identity: HdContractionBackendIdentity | None = None,
    loaded_backend_token: LoadedHdContractionBackendToken | None = None,
) -> torch.Tensor:
    """Write one planned BMM result without mutating ambient matmul policy."""
    _require_precision(precision)
    _record_hd_contraction(left, right, out)
    with _record_hd_stage("hd.contraction.validation"):
        if precision == "fp32_ieee":
            pass
        elif left.dtype != torch.bfloat16 or right.dtype != torch.bfloat16:
            raise TypeError("bf16_tensorcore operands must be bfloat16")
        elif out.dtype != torch.float32:
            raise TypeError("bf16_tensorcore output must be float32")
        elif backend_identity is None or loaded_backend_token is None:
            raise RuntimeError(
                "BF16 contraction requires both backend identity and loaded token"
            )
        else:
            _validate_loaded_token_for_contraction(
                backend_identity,
                loaded_backend_token,
                left,
                right,
                out,
            )
    with _record_hd_bmm_stage(left, right, out):
        if precision == "fp32_ieee":
            return torch.bmm(left, right, out=out)
        assert loaded_backend_token is not None
        loaded_backend_token.operator(left, right, out=out)
        return out


def _bmm_add_fp32(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    destination: torch.Tensor,
    scratch: torch.Tensor,
    precision: str,
    backend_identity: HdContractionBackendIdentity | None = None,
    loaded_backend_token: LoadedHdContractionBackendToken | None = None,
) -> torch.Tensor:
    """Accumulate one BMM through an explicitly planned FP32 scratch."""
    if destination.shape != scratch.shape:
        raise ValueError("contraction destination and scratch shapes must match")
    if precision == "bf16_tensorcore" and destination.dtype != torch.float32:
        raise TypeError("bf16_tensorcore destination must be float32")
    _bmm_fp32(
        left,
        right,
        out=scratch,
        precision=precision,
        backend_identity=backend_identity,
        loaded_backend_token=loaded_backend_token,
    )
    destination.add_(scratch)
    return destination


__all__: tuple[str, ...] = ()
