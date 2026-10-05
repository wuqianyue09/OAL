"""Public causal OAL entry point for HD Block-GEMM."""

from .hd_block_gemm_adapters import grouped_causal_attention

__all__ = ("grouped_causal_attention",)
