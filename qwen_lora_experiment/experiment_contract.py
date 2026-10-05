"""Legacy Qwen constants retained for existing single-model entry points.

New cross-model code uses ``backbones.spec`` for instance geometry and scope.
These aliases preserve the historical Qwen API, formal replacement default,
and best-adapter checkpoint name without importing Torch or the operator package.
"""

from .protocol import FORMAL_MASTER_SEEDS
from .backbones.spec import geometry_for_profile

_LEGACY_QWEN_GEOMETRY = geometry_for_profile("qwen2_5_0_5b")
NUM_LAYERS = _LEGACY_QWEN_GEOMETRY.num_layers
NUM_QUERY_HEADS = _LEGACY_QWEN_GEOMETRY.num_query_heads
NUM_KEY_VALUE_HEADS = _LEGACY_QWEN_GEOMETRY.num_kv_heads
HEAD_DIM = _LEGACY_QWEN_GEOMETRY.head_dim
HIDDEN_SIZE = _LEGACY_QWEN_GEOMETRY.hidden_size
ALL_LORA_LAYER_IDS = tuple(range(NUM_LAYERS))
REPLACEMENT_LAYER_IDS = tuple(range(3, 21))
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
BEST_ADAPTER_FILENAME = "best_adapter.pt"
