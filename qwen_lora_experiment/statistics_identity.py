"""Versioned endpoint randomness retained in OAL checkpoint identities."""

from __future__ import annotations
from .protocol import (
    BOOTSTRAP_RESAMPLES,
    STATISTICS_PROTOCOL_VERSION,
    derive_endpoint_bootstrap_seed,
)


def statistics_identity_for_version(version: str) -> dict[str, object]:
    """Reconstruct the fixed endpoint seed identity for a protocol generation."""
    return {
        "statistics_protocol_version": version,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "endpoint_bootstrap_seeds": {
            endpoint: derive_endpoint_bootstrap_seed(
                statistics_protocol_version=version, endpoint_namespace=endpoint
            )
            for endpoint in ("test_nll", "piqa_acc_norm")
        },
    }


def statistics_identity() -> dict[str, object]:
    return statistics_identity_for_version(STATISTICS_PROTOCOL_VERSION)
