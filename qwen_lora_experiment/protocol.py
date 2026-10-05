"""Versioned, domain-separated randomness for formal LoRA comparisons.

The formal comparison schedule is intentionally small and immutable.  Every
random stream is derived with canonical JSON plus SHA-256 so values do not
depend on Python's randomized ``hash()`` implementation or process state.
"""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import random
from collections.abc import Mapping
from typing import Iterator

SEED_PROTOCOL_VERSION = "qwen_lora_seed_protocol_v1"
STATISTICS_PROTOCOL_VERSION = "oal-qwen-lora-v1"
LEGACY_STATISTICS_PROTOCOL_VERSIONS: tuple[str, ...] = ()
REGISTERED_STATISTICS_PROTOCOL_VERSIONS: tuple[str, ...] = (
    STATISTICS_PROTOCOL_VERSION,
) + LEGACY_STATISTICS_PROTOCOL_VERSIONS
FORMAL_MASTER_SEEDS: tuple[int, ...] = (17, 42, 73)
BOOTSTRAP_RESAMPLES = 10000
_BOOTSTRAP_ENDPOINT_NAMESPACES = frozenset(("test_nll", "piqa_acc_norm"))
_SEED_DOMAINS = frozenset(
    ("lora_init", "training_global_rng", "data_permutation", "diagnostic_inputs")
)
_DERIVATION_FIELDS = frozenset(
    (
        "seed_protocol_version",
        "master_seed",
        "lora_init_seed",
        "training_global_rng_seed",
        "data_permutation_seed",
        "diagnostic_inputs_seed",
    )
)


@dataclass(frozen=True)
class SeedDerivations:
    """The complete persisted random-stream schedule for one master seed."""

    seed_protocol_version: str
    master_seed: int
    lora_init_seed: int
    training_global_rng_seed: int
    data_permutation_seed: int
    diagnostic_inputs_seed: int

    def to_dict(self) -> dict[str, object]:
        """Return only JSON scalar values in a deterministic field order."""
        return dict(asdict(self))


def require_formal_master_seed(master_seed: int) -> int:
    """Reject any master seed outside the registered formal comparison set."""
    if type(master_seed) is not int or master_seed not in FORMAL_MASTER_SEEDS:
        raise ValueError(
            f"master_seed must be one of the formal schedule {FORMAL_MASTER_SEEDS}"
        )
    return master_seed


def derive_seed(master_seed: int, domain: str) -> int:
    """Derive one non-negative scalar seed from an immutable domain label."""
    master_seed = require_formal_master_seed(master_seed)
    if not isinstance(domain, str) or domain not in _SEED_DOMAINS:
        raise ValueError(f"seed domain must be one of {tuple(sorted(_SEED_DOMAINS))}")
    payload = json.dumps(
        {
            "domain": domain,
            "master_seed": master_seed,
            "seed_protocol_version": SEED_PROTOCOL_VERSION,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (1 << 63) - 1


def resolve_seed_derivations(master_seed: int) -> SeedDerivations:
    """Resolve every random stream used by a formal replicate."""
    master_seed = require_formal_master_seed(master_seed)
    return SeedDerivations(
        seed_protocol_version=SEED_PROTOCOL_VERSION,
        master_seed=master_seed,
        lora_init_seed=derive_seed(master_seed, "lora_init"),
        training_global_rng_seed=derive_seed(master_seed, "training_global_rng"),
        data_permutation_seed=derive_seed(master_seed, "data_permutation"),
        diagnostic_inputs_seed=derive_seed(master_seed, "diagnostic_inputs"),
    )


def derive_endpoint_bootstrap_seed(
    *, statistics_protocol_version: str, endpoint_namespace: str
) -> int:
    """Derive an endpoint-scoped fixed bootstrap stream for the formal cohort.

    Bootstrap randomness is deliberately independent from individual training
    replicates.  Its identity is instead the ordered registered schedule, the
    immutable statistics protocol version, and the endpoint namespace.  This
    makes the seed reproducible across processes without permitting callers to
    tune it per report.
    """
    if (
        not isinstance(statistics_protocol_version, str)
        or not statistics_protocol_version
    ):
        raise ValueError("statistics_protocol_version must be a non-empty string")
    if endpoint_namespace not in _BOOTSTRAP_ENDPOINT_NAMESPACES:
        raise ValueError(
            f"endpoint_namespace must be one of {tuple(sorted(_BOOTSTRAP_ENDPOINT_NAMESPACES))}"
        )
    payload = json.dumps(
        {
            "endpoint_namespace": endpoint_namespace,
            "formal_master_seed_schedule": list(FORMAL_MASTER_SEEDS),
            "statistics_protocol_version": statistics_protocol_version,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (1 << 63) - 1


def validate_persisted_seed_derivations(
    value: object, *, master_seed: int
) -> SeedDerivations:
    """Fail closed unless persisted seed evidence is the canonical schedule."""
    expected = resolve_seed_derivations(master_seed)
    if not isinstance(value, Mapping) or set(value) != _DERIVATION_FIELDS:
        raise ValueError(
            "seed_derivations must contain exactly the seed protocol fields"
        )
    actual = dict(value)
    if any(
        (
            type(item) is not int
            for (name, item) in actual.items()
            if name != "seed_protocol_version"
        )
    ):
        raise ValueError("seed_derivations seed values must be integers")
    if actual != expected.to_dict():
        raise ValueError("seed_derivations do not match the formal seed protocol")
    return expected


def seed_global_rng(seed: int) -> None:
    """Seed Python, NumPy, Torch CPU, and every visible Torch CUDA generator.

    Imports are local so configuration parsing remains dependency-free on data
    preparation machines.  NumPy's legacy global MT19937 accepts 32-bit seeds,
    so it receives the low 32 bits of the persisted scalar while Python and
    Torch receive the full domain-derived value.
    """
    if type(seed) is not int or not 0 <= seed < 1 << 63:
        raise ValueError("global RNG seed must be a non-negative 63-bit integer")
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed & (1 << 32) - 1)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_global_rng_state() -> dict[str, object]:
    """Clone every process-global RNG stream used by the pilot runtime."""
    import numpy as np
    import torch

    numpy_state = np.random.get_state()
    cuda_states: tuple[object, ...] = ()
    if torch.cuda.is_available():
        cuda_states = tuple(
            (
                torch.cuda.get_rng_state(device=index).clone()
                for index in range(torch.cuda.device_count())
            )
        )
    return {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": cuda_states,
    }


def restore_global_rng_state(state: Mapping[str, object]) -> None:
    """Restore a snapshot from :func:`capture_global_rng_state` exactly."""
    import numpy as np
    import torch

    if not isinstance(state, Mapping) or set(state) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }:
        raise ValueError("global RNG state has an invalid schema")
    python_state = state["python"]
    numpy_state = state["numpy"]
    torch_cpu_state = state["torch_cpu"]
    torch_cuda_states = state["torch_cuda"]
    if not isinstance(numpy_state, tuple) or len(numpy_state) != 5:
        raise ValueError("global NumPy RNG state has an invalid schema")
    if (
        not isinstance(torch_cpu_state, torch.Tensor)
        or torch_cpu_state.dtype is not torch.uint8
    ):
        raise ValueError("global Torch CPU RNG state must be a uint8 tensor")
    if not isinstance(torch_cuda_states, tuple):
        raise ValueError("global Torch CUDA RNG state must be a tuple")
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_cpu_state)
    expected_cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(torch_cuda_states) != expected_cuda_count:
        raise ValueError(
            "global Torch CUDA RNG state does not match visible device count"
        )
    for index, cuda_state in enumerate(torch_cuda_states):
        if (
            not isinstance(cuda_state, torch.Tensor)
            or cuda_state.dtype is not torch.uint8
        ):
            raise ValueError("global Torch CUDA RNG state must contain uint8 tensors")
        torch.cuda.set_rng_state(cuda_state, device=index)


@contextmanager
def preserve_global_rng_state() -> Iterator[dict[str, object]]:
    """Restore all global RNG streams on both normal and exceptional exits."""
    snapshot = capture_global_rng_state()
    try:
        yield snapshot
    finally:
        restore_global_rng_state(snapshot)


@contextmanager
def scoped_global_rng(seed: int) -> Iterator[None]:
    """Run a block under one derived seed without leaking global RNG changes."""
    with preserve_global_rng_state():
        seed_global_rng(seed)
        yield
