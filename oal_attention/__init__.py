"""Public API for OAL grouped quadratic attention."""

from .grouped_quadratic import oal_attention
from .grouped_quadratic_admission import (
    GroupedExecutionAdmission,
    GroupedExecutionAdmissionError,
    GroupedExecutionAdmissionRequest,
    GroupedExecutionAdmissionStage,
    resolve_grouped_execution_admission,
)
from .grouped_observability import (
    GroupedExecutionBackwardEnvelope,
    GroupedExecutionPhysicalStage,
    GroupedExecutionTrace,
    capture_grouped_execution_trace,
    reactivate_grouped_execution_trace,
)

__all__ = [
    "GroupedExecutionAdmission",
    "GroupedExecutionAdmissionError",
    "GroupedExecutionAdmissionRequest",
    "GroupedExecutionAdmissionStage",
    "GroupedExecutionBackwardEnvelope",
    "GroupedExecutionPhysicalStage",
    "GroupedExecutionTrace",
    "capture_grouped_execution_trace",
    "oal_attention",
    "reactivate_grouped_execution_trace",
    "resolve_grouped_execution_admission",
]
