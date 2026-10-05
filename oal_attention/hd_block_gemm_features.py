"""Private facade for plan-bound FP32 packed features and gradient folds.

These helpers are valid only inside a custom autograd Function's no-grad
forward/backward regions. Pair adjoints use controlled scatter reductions;
CUDA deterministic-algorithm mode is rejected by the fold implementation.
"""

from .hd_block_gemm_cache import (
    _materialize_pair_layout,
    _release_pair_cache_admission_lease,
)
from .hd_block_gemm_feature_backward import (
    _begin_query_coefficient_gradient_accumulation,
    _finalize_query_coefficient_gradients,
    _fold_key_feature_gradient,
    _fold_query_feature_gradient,
)
from .hd_block_gemm_feature_context import (
    _PreparedFeatureContext,
    _copy_into,
    _new_coefficient_coverage,
    _planned_empty,
    _prepare_feature_context,
    _require_scatter_mode,
)
from .hd_block_gemm_feature_forward import (
    _build_key_features,
    _build_query_features,
)

__all__: tuple[str, ...] = ()
