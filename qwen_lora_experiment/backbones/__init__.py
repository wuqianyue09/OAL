"""Dependency-free model profile API and family-specific backbone adapters."""

from .spec import (
    ExperimentScope,
    ModelGeometry,
    ModelProfile,
    geometry_for_profile,
    make_scope,
    validate_runtime_geometry,
)

__all__ = [
    "ExperimentScope",
    "ModelGeometry",
    "ModelProfile",
    "geometry_for_profile",
    "make_scope",
    "validate_runtime_geometry",
]
