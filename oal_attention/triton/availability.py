"""Optional Triton availability, independent of any attention kernel."""

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def triton_is_available() -> bool:
    """Return whether both Triton and its language module can be imported."""
    return triton is not None and tl is not None
