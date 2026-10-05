"""Materialized wave primitives for the private HD Block-GEMM backward."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager

import torch

from .hd_block_gemm_contractions import _bmm_fp32
from .hd_block_gemm_profiling import (
    _QUERY_DPHI_GLOBAL_STAGE,
    _QUERY_DPHI_LOCAL_STAGE,
    _QUERY_DPHI_MERGE_STAGE,
    _QUERY_DS_STAGE,
    _record_hd_bmm_stage,
    _record_hd_contraction,
    _record_hd_stage,
)
from .hd_cublas_compat import (
    HdContractionBackendIdentity,
    LoadedHdContractionBackendToken,
)

_BmmFp32 = Callable[..., torch.Tensor]
_StageRecorder = Callable[[str], AbstractContextManager[object]]


def _expand_kv_block_wave(
    source: torch.Tensor,
    storage: torch.Tensor,
    *,
    gqa_ratio: int,
) -> torch.Tensor:
    """Broadcast [B,Hkv,W,T,C] into the supplied query-head wave storage."""
    batch, kv_heads, blocks, tokens, channels = source.shape
    grouped = storage.view(batch, kv_heads, gqa_ratio, blocks, tokens, channels)
    grouped.copy_(source.unsqueeze(2))
    return grouped.view(batch, kv_heads * gqa_ratio, blocks, tokens, channels)


def _reduce_gqa_block_wave(
    source: torch.Tensor,
    destination: torch.Tensor,
    *,
    gqa_ratio: int,
) -> None:
    """Sum adjacent query heads into a supplied [B,Hkv,W,T,C] FP32 wave."""
    batch, kv_heads, blocks, tokens, channels = destination.shape
    torch.sum(
        source.view(batch, kv_heads, gqa_ratio, blocks, tokens, channels),
        dim=2,
        dtype=torch.float32,
        out=destination,
    )


def _get_dt_wave(
    dt: torch.Tensor,
    *,
    block_start: int,
    wave_blocks: int,
    dt_wave_tc: torch.Tensor | None,
    precision: str,
) -> tuple[torch.Tensor, int]:
    """Select FP32 dT or round its active block range into reusable BF16 storage."""
    if precision != "bf16_tensorcore":
        return dt, block_start
    if dt_wave_tc is None:
        raise RuntimeError("Tensor-Core dT wave storage was not initialized")
    dt_wave_tc[:, :, :wave_blocks].copy_(
        dt[:, :, block_start : block_start + wave_blocks]
    )
    if wave_blocks < dt_wave_tc.shape[2]:
        dt_wave_tc[:, :, wave_blocks:].zero_()
    return dt_wave_tc, 0


def _stage_full_bf16_gradient(
    augmented_gradient: torch.Tensor,
    *,
    cache: torch.Tensor,
    token_block: int,
    feature_wave_blocks: int,
) -> torch.Tensor:
    """Round FP32 normalization output once into padded per-wave BF16 storage."""
    if augmented_gradient.ndim != 4 or augmented_gradient.dtype != torch.float32:
        raise ValueError("full gradient staging requires rank-four FP32 input")
    if cache.ndim != 6 or cache.dtype != torch.bfloat16:
        raise ValueError("full gradient staging requires rank-six BF16 cache")
    if token_block <= 0 or feature_wave_blocks <= 0:
        raise ValueError("full gradient staging geometry must be positive")
    batch_size, query_heads, sequence_length, channels = augmented_gradient.shape
    (
        number_waves,
        cache_batch,
        cache_heads,
        wave_blocks,
        block_tokens,
        cache_channels,
    ) = cache.shape
    if (
        cache_batch != batch_size
        or cache_heads != query_heads
        or wave_blocks != feature_wave_blocks
        or block_tokens != token_block
        or cache_channels != channels
    ):
        raise ValueError("full gradient staging cache shape does not match")
    wave_tokens = feature_wave_blocks * token_block
    expected_waves = (sequence_length + wave_tokens - 1) // wave_tokens
    if number_waves != expected_waves:
        raise ValueError("full gradient staging wave count does not match")
    if cache.device != augmented_gradient.device:
        raise ValueError("full gradient staging cache device does not match")

    for wave_index in range(number_waves):
        token_start = wave_index * wave_tokens
        token_end = min(sequence_length, token_start + wave_tokens)
        valid_tokens = token_end - token_start
        destination = cache[wave_index].view(
            batch_size,
            query_heads,
            wave_tokens,
            channels,
        )
        destination[:, :, :valid_tokens].copy_(
            augmented_gradient[:, :, token_start:token_end]
        )
        if valid_tokens < wave_tokens:
            destination[:, :, valid_tokens:].zero_()
    return cache


def _get_g_wave(
    augmented_gradient: torch.Tensor,
    *,
    token_start: int,
    token_end: int,
    wave_index: int,
    g_wave: torch.Tensor | None,
    full_bf16_cache: torch.Tensor | None,
) -> torch.Tensor:
    """Return one dense BF16/FP32 wave from the selected staging policy."""
    if (g_wave is None) == (full_bf16_cache is None):
        raise ValueError("backward gradient requires exactly one staging storage")
    if full_bf16_cache is not None:
        if full_bf16_cache.ndim != 6 or full_bf16_cache.dtype != torch.bfloat16:
            raise ValueError("full backward gradient cache is invalid")
        if wave_index < 0 or wave_index >= full_bf16_cache.shape[0]:
            raise ValueError("backward gradient wave index is invalid")
        wave = full_bf16_cache[wave_index].view(
            full_bf16_cache.shape[1],
            full_bf16_cache.shape[2],
            full_bf16_cache.shape[3] * full_bf16_cache.shape[4],
            full_bf16_cache.shape[5],
        )
        expected_start = wave_index * wave.shape[2]
        if token_start != expected_start or token_end < token_start:
            raise ValueError("backward gradient cache range does not match its wave")
        if token_end > augmented_gradient.shape[2]:
            raise ValueError("backward gradient wave exceeds the gradient")
        return wave

    assert g_wave is not None
    _prepare_wave_inputs(
        augmented_gradient,
        None,
        token_start=token_start,
        token_end=token_end,
        g_wave=g_wave,
        u_wave=None,
        value_dimension=augmented_gradient.shape[-1] - 1,
    )
    return g_wave


def _prepare_wave_inputs(
    augmented_gradient: torch.Tensor | None,
    values: torch.Tensor | None,
    *,
    token_start: int,
    token_end: int,
    g_wave: torch.Tensor | None,
    u_wave: torch.Tensor | None,
    value_dimension: int,
) -> int:
    """Copy one active token range into reusable g/U wave storage."""
    if token_start < 0 or token_end < token_start:
        raise ValueError("backward wave token range is invalid")
    valid_tokens = token_end - token_start
    if g_wave is not None:
        capacity_tokens = g_wave.shape[2]
    elif u_wave is not None:
        capacity_tokens = u_wave.shape[2]
    else:
        raise ValueError("backward wave requires g or U storage")
    if valid_tokens > capacity_tokens:
        raise ValueError("backward wave exceeds its planned token capacity")
    if (augmented_gradient is None) != (g_wave is None):
        raise ValueError("backward gradient input and wave storage must be paired")
    if augmented_gradient is not None and g_wave is not None:
        if token_end > augmented_gradient.shape[2]:
            raise ValueError("backward wave token range exceeds the gradient")
        if tuple(g_wave.shape[:2]) != tuple(augmented_gradient.shape[:2]):
            raise ValueError("backward gradient wave head shape does not match")
        if g_wave.shape[-1] != augmented_gradient.shape[-1]:
            raise ValueError("backward gradient wave channel shape does not match")
        g_wave[:, :, :valid_tokens, :].copy_(
            augmented_gradient[:, :, token_start:token_end, :]
        )
        if valid_tokens < capacity_tokens:
            g_wave[:, :, valid_tokens:, :].zero_()

    if (values is None) != (u_wave is None):
        raise ValueError("backward value input and wave storage must be paired")
    if values is not None and u_wave is not None:
        if token_end > values.shape[2]:
            raise ValueError("backward wave token range exceeds the values")
        if u_wave.shape[2] != capacity_tokens:
            raise ValueError("backward g/U wave capacities do not match")
        if tuple(u_wave.shape[:2]) != tuple(values.shape[:2]):
            raise ValueError("backward value wave head shape does not match")
        if value_dimension != values.shape[-1]:
            raise ValueError("backward value dimension does not match")
        if u_wave.shape[-1] != value_dimension + 1:
            raise ValueError("backward augmented value dimension does not match")
        u_wave[:, :, :valid_tokens, :value_dimension].copy_(
            values[:, :, token_start:token_end, :]
        )
        u_wave[:, :, :valid_tokens, value_dimension].fill_(1.0)
        if valid_tokens < capacity_tokens:
            u_wave[:, :, valid_tokens:, :].zero_()
    return valid_tokens


def _require_precomputed_ds(
    precomputed_ds: torch.Tensor,
    *,
    ds: torch.Tensor,
    precision: str,
) -> torch.Tensor:
    if tuple(precomputed_ds.shape) != tuple(ds.shape):
        raise ValueError("precomputed backward dS shape does not match")
    expected_dtype = torch.bfloat16 if precision == "bf16_tensorcore" else ds.dtype
    if precomputed_ds.dtype != expected_dtype:
        raise TypeError("precomputed backward dS dtype does not match")
    if precomputed_ds.device != ds.device:
        raise ValueError("precomputed backward dS device does not match")
    return precomputed_ds


def _compute_local_ds(
    g: torch.Tensor,
    u: torch.Tensor,
    *,
    ds: torch.Tensor,
    ds_tc: torch.Tensor | None,
    precision: str,
    backend_identity: HdContractionBackendIdentity | None,
    loaded_backend_token: LoadedHdContractionBackendToken | None,
    stage: str = _QUERY_DS_STAGE,
    bmm_fp32: _BmmFp32 = _bmm_fp32,
    record_stage: _StageRecorder = _record_hd_stage,
) -> torch.Tensor:
    """Produce the rounded lower-triangular local dS operand once."""
    if g.ndim != 3 or u.ndim != 3 or ds.ndim != 3:
        raise ValueError("backward wave contractions require rank-three tensors")
    if g.shape != u.shape:
        raise ValueError("backward wave g/U shapes do not match")
    expected_ds_shape = (g.shape[0], g.shape[1], u.shape[1])
    if tuple(ds.shape) != expected_ds_shape:
        raise ValueError("backward wave dS shape does not match")
    with record_stage(stage):
        bmm_fp32(
            g,
            u.transpose(1, 2),
            out=ds,
            precision=precision,
            backend_identity=backend_identity,
            loaded_backend_token=loaded_backend_token,
        )
    ds.tril_()
    if precision == "bf16_tensorcore":
        if ds_tc is None:
            raise RuntimeError("Tensor-Core dS storage was not initialized")
        if tuple(ds_tc.shape) != tuple(ds.shape) or ds_tc.dtype != torch.bfloat16:
            raise ValueError("Tensor-Core dS storage does not match its producer")
        ds_tc.copy_(ds)
        return ds_tc
    if precision != "fp32_ieee":
        raise ValueError("unsupported HD Block-GEMM contraction precision")
    return ds


def _query_wave_vjp(
    g: torch.Tensor,
    u: torch.Tensor,
    phi_k: torch.Tensor,
    carry: torch.Tensor,
    *,
    ds: torch.Tensor,
    ds_tc: torch.Tensor | None,
    d_phi_q: torch.Tensor,
    d_phi_q_local: torch.Tensor | None,
    precomputed_ds: torch.Tensor | None = None,
    precision: str,
    backend_identity: HdContractionBackendIdentity | None,
    loaded_backend_token: LoadedHdContractionBackendToken | None,
    bmm_fp32: _BmmFp32 = _bmm_fp32,
    record_stage: _StageRecorder = _record_hd_stage,
) -> torch.Tensor:
    """Consume one materialized query wave, optionally reusing its dS."""
    ds_operand = (
        _compute_local_ds(
            g,
            u,
            ds=ds,
            ds_tc=ds_tc,
            precision=precision,
            backend_identity=backend_identity,
            loaded_backend_token=loaded_backend_token,
            stage=_QUERY_DS_STAGE,
            bmm_fp32=bmm_fp32,
            record_stage=record_stage,
        )
        if precomputed_ds is None
        else _require_precomputed_ds(
            precomputed_ds,
            ds=ds,
            precision=precision,
        )
    )
    with record_stage(_QUERY_DPHI_GLOBAL_STAGE):
        bmm_fp32(
            g,
            carry.transpose(1, 2),
            out=d_phi_q,
            precision=precision,
            backend_identity=backend_identity,
            loaded_backend_token=loaded_backend_token,
        )
    if precision == "bf16_tensorcore":
        if d_phi_q_local is None:
            raise RuntimeError("Tensor-Core local dPhiQ storage was not initialized")
        with record_stage(_QUERY_DPHI_LOCAL_STAGE):
            bmm_fp32(
                ds_operand,
                phi_k,
                out=d_phi_q_local,
                precision=precision,
                backend_identity=backend_identity,
                loaded_backend_token=loaded_backend_token,
            )
        with record_stage(_QUERY_DPHI_MERGE_STAGE):
            d_phi_q.add_(d_phi_q_local)
    else:
        with record_stage(_QUERY_DPHI_LOCAL_STAGE):
            _record_hd_contraction(ds_operand, phi_k, d_phi_q)
            with _record_hd_bmm_stage(ds_operand, phi_k, d_phi_q):
                torch.baddbmm(
                    d_phi_q,
                    ds_operand,
                    phi_k,
                    beta=1.0,
                    out=d_phi_q,
                )
    return d_phi_q


def _kv_wave_vjp(
    phi_q: torch.Tensor,
    g: torch.Tensor,
    *,
    u: torch.Tensor | None,
    phi_k: torch.Tensor | None,
    ds: torch.Tensor | None,
    ds_tc: torch.Tensor | None,
    d_phi_k: torch.Tensor | None,
    local_score: torch.Tensor | None,
    local_score_tc: torch.Tensor | None,
    du: torch.Tensor | None,
    saved_local_score_tc: torch.Tensor | None = None,
    precomputed_ds: torch.Tensor | None = None,
    precision: str,
    backend_identity: HdContractionBackendIdentity | None,
    loaded_backend_token: LoadedHdContractionBackendToken | None,
    bmm_fp32: _BmmFp32 = _bmm_fp32,
    record_stage: _StageRecorder = _record_hd_stage,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Produce one KV wave's query-head-local dPhiK and dU terms."""
    if d_phi_k is not None:
        if u is None or ds is None:
            raise RuntimeError("dK wave dS inputs were not initialized")
        ds_operand = (
            _compute_local_ds(
                g,
                u,
                ds=ds,
                ds_tc=ds_tc,
                precision=precision,
                backend_identity=backend_identity,
                loaded_backend_token=loaded_backend_token,
                stage="hd.backward.kv_contractions",
                bmm_fp32=bmm_fp32,
                record_stage=record_stage,
            )
            if precomputed_ds is None
            else _require_precomputed_ds(
                precomputed_ds,
                ds=ds,
                precision=precision,
            )
        )
        with record_stage("hd.backward.kv_contractions"):
            bmm_fp32(
                ds_operand.transpose(1, 2),
                phi_q,
                out=d_phi_k,
                precision=precision,
                backend_identity=backend_identity,
                loaded_backend_token=loaded_backend_token,
            )

    if du is not None:
        if saved_local_score_tc is not None:
            if precision != "bf16_tensorcore":
                raise ValueError("saved local score requires bf16_tensorcore precision")
            if local_score is not None or local_score_tc is not None:
                raise ValueError("saved local score must replace recompute storage")
            expected_score_shape = (g.shape[0], g.shape[1], g.shape[1])
            if (
                tuple(saved_local_score_tc.shape) != expected_score_shape
                or saved_local_score_tc.dtype != torch.bfloat16
                or saved_local_score_tc.device != g.device
                or not saved_local_score_tc.is_contiguous()
            ):
                raise ValueError("saved local score does not match the dV wave")
            score_operand = saved_local_score_tc
        else:
            if phi_k is None or local_score is None:
                raise RuntimeError("dV wave score inputs were not initialized")
            with record_stage("hd.backward.kv_contractions"):
                bmm_fp32(
                    phi_q,
                    phi_k.transpose(1, 2),
                    out=local_score,
                    precision=precision,
                    backend_identity=backend_identity,
                    loaded_backend_token=loaded_backend_token,
                )
            local_score.tril_()
            score_operand = local_score
            if precision == "bf16_tensorcore":
                if local_score_tc is None:
                    raise RuntimeError(
                        "Tensor-Core local score storage was not initialized"
                    )
                if (
                    tuple(local_score_tc.shape) != tuple(local_score.shape)
                    or local_score_tc.dtype != torch.bfloat16
                ):
                    raise ValueError(
                        "Tensor-Core local score storage does not match its producer"
                    )
                local_score_tc.copy_(local_score)
                score_operand = local_score_tc
        with record_stage("hd.backward.kv_contractions"):
            bmm_fp32(
                score_operand.transpose(1, 2),
                g,
                out=du,
                precision=precision,
                backend_identity=backend_identity,
                loaded_backend_token=loaded_backend_token,
            )
    return d_phi_k, du


__all__: tuple[str, ...] = ()
