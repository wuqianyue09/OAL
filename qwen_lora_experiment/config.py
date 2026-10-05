"""Immutable, dependency-free configuration for the Qwen LoRA pilot.

The pilot intentionally accepts one narrow experiment profile.  Keeping this
module standard-library only makes local configuration validation possible
without model, dataset, or CUDA packages installed.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping, cast
from .backbones.spec import (
    ExperimentScope,
    ModelGeometry,
    ModelProfile,
    geometry_for_profile,
    make_scope,
)
from .experiment_contract import (
    FORMAL_MASTER_SEEDS,
    LORA_TARGET_MODULES,
    NUM_LAYERS,
    REPLACEMENT_LAYER_IDS,
)
from .protocol import (
    SEED_PROTOCOL_VERSION,
    SeedDerivations,
    resolve_seed_derivations,
    validate_persisted_seed_derivations,
)

MethodName = Literal["grouped_quadratic",]
TuningMode = Literal["lora"]
DTypeName = Literal["bfloat16"]
GroupedAssetMode = Literal["precomputed"]
AttentionBackend = Literal["legacy", "hd_block_gemm"]
HDPrecision = Literal["fp32_ieee", "bf16_tensorcore"]
GROUPED_BASE_METHODS: frozenset[MethodName] = frozenset(("grouped_quadratic",))
TRAINABLE_KERNEL_METHODS: frozenset[MethodName] = frozenset(("grouped_quadratic",))
METHOD_NAMES: tuple[MethodName, ...] = ("grouped_quadratic",)


def method_identity_for_profile(identity: str, profile: ModelProfile) -> str:
    """Return a model-factual method label while preserving legacy Qwen bytes."""
    suffix = "_causal_qwen"
    if profile == "llama3_2_1b_base" and identity.endswith(suffix):
        return identity[: -len(suffix)] + "_causal_llama"
    return identity


FORMAL_ASSET_SHA256_TEMPLATE = "0" * 64
GROUPED_ASSET_MODES: tuple[GroupedAssetMode, ...] = ("precomputed",)
DEFAULT_TRAIN_TOKEN_BUDGET = 2097152
DEFAULT_SEQUENCE_LENGTH = 2048
FALLBACK_SEQUENCE_LENGTH = 4096
DEFAULT_GROUPED_WORKSPACE_BUDGET_BYTES = 512 * 2**20
DEFAULT_LORA_TARGETS = LORA_TARGET_MODULES
MODEL_NUM_LAYERS = NUM_LAYERS
DEFAULT_REPLACEMENT_LAYERS = REPLACEMENT_LAYER_IDS
PRE_REGISTERED_MASTER_SEEDS = FORMAL_MASTER_SEEDS
_EFFECTIVE_CONFIG_DERIVED_FIELDS = frozenset(
    {
        "replaced_layer_ids_zero_based",
        "trainable_kernel_layer_ids",
        "expected_group_asset_status",
        "group_asset_sequence_length",
        "method_identity",
        "seed_protocol_version",
        "seed_derivations",
    }
)
_OPTIONAL_ATTENTION_FIELDS = frozenset({"attention_backend", "hd_options"})
_OPTIONAL_PROFILE_FIELDS = frozenset({"model_profile"})
_HD_METHODS = frozenset({"grouped_quadratic"})


@dataclass(frozen=True)
class HDOptions:
    """Finite execution settings passed to the operator HD planner."""

    precision: HDPrecision = "bf16_tensorcore"
    token_block: int = 64
    feature_wave_blocks: int = 4
    memory_budget_bytes: int | None = None
    key_feature_impl: str | None = None
    key_fold_impl: str | None = None
    query_feature_impl: str | None = None
    query_feature_token_tile: int | None = None
    query_fold_impl: str | None = None
    query_fold_token_tile: int | None = None
    query_fold_input: str = "staged_fp32"
    query_gradient_flow: str = "materialized"
    backward_schedule: str | None = None
    gradient_staging: str = "per_wave"
    forward_normalize_impl: str | None = None
    backward_normalize_impl: str | None = None
    kv_cross_impl: str | None = None
    save_local_score: bool | None = None
    key_retention: str | None = None
    feature_padding: str = "none"

    def resolved(self) -> HDOptions:
        if self.precision == "bf16_tensorcore":
            defaults: dict[str, object] = {
                "key_feature_impl": "triton_materialized",
                "key_fold_impl": "triton_materialized",
                "query_feature_impl": "triton_materialized",
                "query_feature_token_tile": 4,
                "query_fold_impl": "triton_materialized",
                "query_fold_token_tile": 1,
                "backward_schedule": "shared_wave",
                "forward_normalize_impl": "triton",
                "backward_normalize_impl": "triton",
                "kv_cross_impl": "batched_dense",
                "save_local_score": True,
                "key_retention": "forward",
            }
        else:
            defaults = {
                "key_feature_impl": "generic_materialized",
                "key_fold_impl": "generic_materialized",
                "query_feature_impl": "generic_materialized",
                "query_feature_token_tile": 1,
                "query_fold_impl": "generic_materialized",
                "query_fold_token_tile": 1,
                "backward_schedule": "split",
                "forward_normalize_impl": "torch",
                "backward_normalize_impl": "torch",
                "kv_cross_impl": "split",
                "save_local_score": False,
                "key_retention": "none",
            }
        values = asdict(self)
        for name, value in defaults.items():
            if values[name] is None:
                values[name] = value
        resolved = HDOptions(**values)
        resolved.validate(require_resolved=True)
        return resolved

    def validate(self, *, require_resolved: bool = False) -> None:
        if self.precision not in {"fp32_ieee", "bf16_tensorcore"}:
            raise ValueError(
                "hd_options.precision must be fp32_ieee or bf16_tensorcore"
            )
        _require_positive_int("hd_options.token_block", self.token_block)
        _require_positive_int(
            "hd_options.feature_wave_blocks", self.feature_wave_blocks
        )
        if self.memory_budget_bytes is not None:
            _require_positive_int(
                "hd_options.memory_budget_bytes", self.memory_budget_bytes
            )
        if require_resolved and any(
            (
                getattr(self, name) is None
                for name in (
                    "key_feature_impl",
                    "key_fold_impl",
                    "query_feature_impl",
                    "query_feature_token_tile",
                    "query_fold_impl",
                    "query_fold_token_tile",
                    "backward_schedule",
                    "forward_normalize_impl",
                    "backward_normalize_impl",
                    "kv_cross_impl",
                    "save_local_score",
                    "key_retention",
                )
            )
        ):
            raise ValueError(
                "resolved hd_options may not contain null planner settings"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self.resolved())


_HD_OPTION_FIELDS = frozenset((field.name for field in fields(HDOptions)))


@dataclass(frozen=True)
class MethodSpec:
    """Static capabilities that distinguish the comparable methods."""

    name: MethodName
    execution: str
    trains_kernel_parameters: bool
    requires_group_asset: bool


def _method_spec(*, name: MethodName, execution: str) -> MethodSpec:
    """Build one registry entry from the canonical method-family sets."""
    return MethodSpec(
        name=name,
        execution=execution,
        trains_kernel_parameters=name in TRAINABLE_KERNEL_METHODS,
        requires_group_asset=name in GROUPED_BASE_METHODS,
    )


METHOD_REGISTRY: Mapping[MethodName, MethodSpec] = MappingProxyType(
    {"grouped_quadratic": _method_spec(name="grouped_quadratic", execution="triton")}
)


@dataclass(frozen=True)
class LoRAConfig:
    """The fixed LoRA injection profile shared by every pilot method."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    targets: tuple[str, ...] = DEFAULT_LORA_TARGETS

    def validate(self) -> None:
        _require_exact("lora.rank", self.rank, 8)
        _require_exact("lora.alpha", self.alpha, 16.0)
        _require_exact("lora.dropout", self.dropout, 0.05)
        if self.targets != DEFAULT_LORA_TARGETS:
            raise ValueError(
                "lora.targets must be exactly ('q_proj', 'k_proj', 'v_proj', 'o_proj')"
            )


@dataclass(frozen=True)
class PilotConfig:
    """Fully specified, shared profile for one LoRA-pilot method run."""

    method: MethodName
    model_path: Path
    data_root: Path
    runs_root: Path
    group_asset_path: Path
    group_asset_sha256: str | None = None
    model_profile: ModelProfile | None = None
    group_asset_mode: GroupedAssetMode = "precomputed"
    attention_backend: AttentionBackend = "legacy"
    hd_options: HDOptions | None = None
    replacement_layers: tuple[int, ...] = DEFAULT_REPLACEMENT_LAYERS
    tuning_mode: TuningMode = "lora"
    sequence_length: int = 4096
    train_token_budget: int = DEFAULT_TRAIN_TOKEN_BUDGET
    validation_blocks: int = 32
    validation_interval_steps: int = 128
    diagnostic_sequence_length: int = 128
    diagnostic_interval_steps: int = 128
    seed: int = 42
    grouped_kernel_epsilon: float = 1e-06
    grouped_workspace_budget_bytes: int = DEFAULT_GROUPED_WORKSPACE_BUDGET_BYTES
    lora_learning_rate: float = 0.0005
    lora_weight_decay: float = 0.01
    lora_beta1: float = 0.9
    lora_beta2: float = 0.999
    lora_epsilon: float = 1e-08
    kernel_learning_rate: float = 0.0005
    kernel_weight_decay: float = 0.01
    kernel_beta1: float = 0.9
    kernel_beta2: float = 0.999
    kernel_epsilon: float = 1e-08
    optimizer_candidate_configurations: int = 1
    optimizer_tuning_seed_count: int = 1
    pre_registered_master_seeds: tuple[int, ...] = PRE_REGISTERED_MASTER_SEEDS
    warmup_ratio: float = 0.05
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    dtype: DTypeName = "bfloat16"
    lora: LoRAConfig = LoRAConfig()

    @property
    def train_blocks(self) -> int:
        if self.train_token_budget % self.sequence_length != 0:
            raise ValueError("train_token_budget must be divisible by sequence_length")
        return self.train_token_budget // self.sequence_length

    @property
    def method_spec(self) -> MethodSpec:
        return METHOD_REGISTRY[self.method]

    @property
    def resolved_model_profile(self) -> ModelProfile:
        return self.model_profile or "qwen2_5_0_5b"

    @property
    def geometry(self) -> ModelGeometry:
        return geometry_for_profile(self.resolved_model_profile)

    @property
    def scope(self) -> ExperimentScope:
        return make_scope(self.geometry, self.replacement_layers)

    @property
    def attention_execution(self) -> str:
        if self.attention_backend == "hd_block_gemm":
            return "hd_block_gemm_causal"
        return self.method_spec.execution

    @property
    def resolved_hd_options(self) -> HDOptions:
        if self.attention_backend != "hd_block_gemm":
            raise ValueError("resolved_hd_options is available only for hd_block_gemm")
        return (self.hd_options or HDOptions()).resolved()

    @property
    def replacement_layer_ids(self) -> tuple[int, ...]:
        """Return the attention backends actually replaced by this method."""
        return self.scope.actual_replacement_layers(self.method)

    @property
    def trainable_kernel_layer_ids(self) -> tuple[int, ...]:
        """Return the active method-specific parameter owners, never LoRA scope."""
        if self.method in TRAINABLE_KERNEL_METHODS:
            return self.replacement_layer_ids
        return ()

    @property
    def uses_grouped_quadratic_base(self) -> bool:
        """Whether this method executes the shared OAL base."""
        return self.method in GROUPED_BASE_METHODS

    @property
    def uses_precomputed_group_asset(self) -> bool:
        """Whether OAL reads the released numerical result instead of a calibration record."""
        return (
            self.method == "grouped_quadratic"
            and self.group_asset_mode == "precomputed"
        )

    @property
    def expected_group_asset_status(self) -> GroupedAssetMode:
        """Return the only canonical asset status allowed for this run."""
        return self.group_asset_mode

    @property
    def group_asset_sequence_length(self) -> int:
        """Return the immutable source sequence length for the asset profile."""
        return self.sequence_length

    @property
    def method_identity(self) -> str:
        """Disambiguate formal Adaptive MultiGroup from a legacy transfer ablation."""
        if self.method != "grouped_quadratic":
            identity = self.method
        else:
            identity = "adaptive_multigroup_full_quadratic"
        return method_identity_for_profile(identity, self.resolved_model_profile)

    def __post_init__(self) -> None:
        if self.group_asset_sha256 == FORMAL_ASSET_SHA256_TEMPLATE:
            object.__setattr__(self, "group_asset_sha256", None)

    @property
    def seed_derivations(self) -> SeedDerivations:
        """Return the versioned random streams for this formal replicate."""
        return resolve_seed_derivations(self.seed)

    def validate(self) -> None:
        geometry = self.geometry
        if self.method not in METHOD_REGISTRY:
            raise ValueError(
                f"method must be one of {METHOD_NAMES}; got {self.method!r}"
            )
        _parse_tuning_mode(self.tuning_mode)
        _require_path("model_path", self.model_path)
        _require_path("data_root", self.data_root)
        _require_path("runs_root", self.runs_root)
        _require_group_asset_path(self.group_asset_path)
        _parse_group_asset_mode(self.group_asset_mode)
        if self.attention_backend not in {"legacy", "hd_block_gemm"}:
            raise ValueError("attention_backend must be legacy or hd_block_gemm")
        if self.attention_backend == "legacy":
            if self.hd_options is not None:
                raise ValueError(
                    "hd_options are allowed only with attention_backend=hd_block_gemm"
                )
        else:
            if self.method not in _HD_METHODS or self.tuning_mode != "lora":
                raise ValueError("hd_block_gemm is allowed only for OAL in lora mode")
            self.resolved_hd_options.validate(require_resolved=True)
        if self.group_asset_sha256 is not None:
            _require_sha256("group_asset_sha256", self.group_asset_sha256)
        if self.group_asset_mode == "precomputed" and self.uses_grouped_quadratic_base:
            if (
                self.method != "grouped_quadratic"
                or self.resolved_model_profile != "qwen2_5_0_5b"
                or self.sequence_length != 4096
            ):
                raise ValueError(
                    "precomputed OAL results support Qwen grouped_quadratic at sequence_length=4096"
                )
        _validate_replacement_layers(
            self.replacement_layers, geometry.num_layers, allow_empty=False
        )
        _require_positive_int("sequence_length", self.sequence_length)
        _require_positive_int("train_token_budget", self.train_token_budget)
        _ = self.train_blocks
        _require_exact("validation_blocks", self.validation_blocks, 32)
        _require_exact("validation_interval_steps", self.validation_interval_steps, 128)
        _require_positive_int(
            "diagnostic_sequence_length", self.diagnostic_sequence_length
        )
        _require_exact("diagnostic_interval_steps", self.diagnostic_interval_steps, 128)
        try:
            derivations = self.seed_derivations
        except ValueError as exc:
            raise ValueError(
                f"seed must be one of the pre_registered_master_seeds {PRE_REGISTERED_MASTER_SEEDS}"
            ) from exc
        _require_positive_finite("grouped_kernel_epsilon", self.grouped_kernel_epsilon)
        _require_positive_int(
            "grouped_workspace_budget_bytes", self.grouped_workspace_budget_bytes
        )
        _require_exact("lora_learning_rate", self.lora_learning_rate, 0.0005)
        _require_exact("lora_weight_decay", self.lora_weight_decay, 0.01)
        _require_exact("lora_beta1", self.lora_beta1, 0.9)
        _require_exact("lora_beta2", self.lora_beta2, 0.999)
        _require_exact("lora_epsilon", self.lora_epsilon, 1e-08)
        _require_exact("kernel_learning_rate", self.kernel_learning_rate, 0.0005)
        _require_exact("kernel_weight_decay", self.kernel_weight_decay, 0.01)
        _require_exact("kernel_beta1", self.kernel_beta1, 0.9)
        _require_exact("kernel_beta2", self.kernel_beta2, 0.999)
        _require_exact("kernel_epsilon", self.kernel_epsilon, 1e-08)
        _require_exact(
            "optimizer_candidate_configurations",
            self.optimizer_candidate_configurations,
            1,
        )
        _require_exact(
            "optimizer_tuning_seed_count", self.optimizer_tuning_seed_count, 1
        )
        _require_exact(
            "pre_registered_master_seeds",
            self.pre_registered_master_seeds,
            PRE_REGISTERED_MASTER_SEEDS,
        )
        _require_exact("warmup_ratio", self.warmup_ratio, 0.05)
        _require_exact("gradient_clip", self.gradient_clip, 1.0)
        _require_exact(
            "gradient_accumulation_steps", self.gradient_accumulation_steps, 1
        )
        _require_exact("dtype", self.dtype, "bfloat16")
        self.lora.validate()
        if self.method_spec.requires_group_asset and (not str(self.group_asset_path)):
            raise ValueError(
                "group_asset_path is required for method grouped_quadratic"
            )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation suitable for an effective config."""
        raw = asdict(self)
        if self.group_asset_sha256 is None:
            raw.pop("group_asset_sha256")
        if self.model_profile is None:
            raw.pop("model_profile")
        if self.attention_backend == "legacy":
            raw.pop("attention_backend")
            raw.pop("hd_options")
        else:
            raw["hd_options"] = self.resolved_hd_options.to_dict()
        for name in ("model_path", "data_root", "runs_root", "group_asset_path"):
            raw[name] = str(raw[name])
        raw["replacement_layers"] = list(self.replacement_layers)
        raw["replaced_layer_ids_zero_based"] = list(self.replacement_layer_ids)
        raw["trainable_kernel_layer_ids"] = list(self.trainable_kernel_layer_ids)
        raw["expected_group_asset_status"] = self.expected_group_asset_status
        raw["group_asset_sequence_length"] = self.group_asset_sequence_length
        raw["method_identity"] = self.method_identity
        derivations = self.seed_derivations
        raw["seed_protocol_version"] = SEED_PROTOCOL_VERSION
        raw["seed_derivations"] = derivations.to_dict()
        raw["pre_registered_master_seeds"] = list(self.pre_registered_master_seeds)
        raw["lora"]["targets"] = list(self.lora.targets)
        return raw


_TOP_LEVEL_FIELDS = frozenset((field.name for field in fields(PilotConfig)))
_LORA_FIELDS = frozenset((field.name for field in fields(LoRAConfig)))


def from_json_file(path: str | Path) -> PilotConfig:
    """Load one fully explicit configuration while rejecting undeclared keys."""
    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as config_file:
            raw = json.load(config_file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"config JSON is invalid: {exc.msg}") from exc
    except OSError as exc:
        raise ValueError(f"config file could not be read: {config_path}") from exc
    return from_mapping(raw)


def from_mapping(raw: object) -> PilotConfig:
    """Construct and validate a config from a JSON-decoded mapping."""
    if not isinstance(raw, Mapping):
        raise ValueError("config must be a JSON object")
    raw = _normalize_historical_effective_config(raw)
    _reject_unknown_and_missing(
        "config",
        raw,
        _TOP_LEVEL_FIELDS,
        optional_fields=_EFFECTIVE_CONFIG_DERIVED_FIELDS
        | _OPTIONAL_ATTENTION_FIELDS
        | _OPTIONAL_PROFILE_FIELDS
        | {"group_asset_sha256"},
    )
    missing_protocol_fields = sorted(
        {"seed_protocol_version", "seed_derivations"} - set(raw)
    )
    if missing_protocol_fields:
        raise ValueError(
            "formal config is missing required seed_protocol evidence: "
            + ", ".join(missing_protocol_fields)
        )
    if raw["seed_protocol_version"] != SEED_PROTOCOL_VERSION:
        raise ValueError(
            "seed_protocol_version does not match the formal seed protocol"
        )
    lora_raw = raw["lora"]
    if not isinstance(lora_raw, Mapping):
        raise ValueError("lora must be an object")
    _reject_unknown_and_missing("lora", lora_raw, _LORA_FIELDS)
    source_seed = _require_int("seed", raw["seed"])
    method = _parse_method(raw["method"])
    model_profile = (
        _parse_model_profile(raw["model_profile"]) if "model_profile" in raw else None
    )
    geometry = geometry_for_profile(model_profile or "qwen2_5_0_5b")
    config = PilotConfig(
        method=method,
        tuning_mode=_parse_tuning_mode(raw["tuning_mode"]),
        model_path=Path(_require_nonempty_string("model_path", raw["model_path"])),
        data_root=Path(_require_nonempty_string("data_root", raw["data_root"])),
        runs_root=Path(_require_nonempty_string("runs_root", raw["runs_root"])),
        group_asset_path=Path(
            _require_nonempty_string("group_asset_path", raw["group_asset_path"])
        ),
        group_asset_sha256=(
            _require_string("group_asset_sha256", raw["group_asset_sha256"])
            if raw.get("group_asset_sha256") is not None
            else None
        ),
        model_profile=model_profile,
        group_asset_mode=_parse_group_asset_mode(raw["group_asset_mode"]),
        attention_backend=_parse_attention_backend(
            raw.get("attention_backend", "legacy")
        ),
        hd_options=_parse_hd_options(raw.get("hd_options")),
        replacement_layers=_parse_replacement_layers(
            raw["replacement_layers"], geometry.num_layers, allow_empty=False
        ),
        sequence_length=_require_int("sequence_length", raw["sequence_length"]),
        train_token_budget=_require_int(
            "train_token_budget", raw["train_token_budget"]
        ),
        validation_blocks=_require_int("validation_blocks", raw["validation_blocks"]),
        validation_interval_steps=_require_int(
            "validation_interval_steps", raw["validation_interval_steps"]
        ),
        diagnostic_sequence_length=_require_int(
            "diagnostic_sequence_length", raw["diagnostic_sequence_length"]
        ),
        diagnostic_interval_steps=_require_int(
            "diagnostic_interval_steps", raw["diagnostic_interval_steps"]
        ),
        seed=source_seed,
        grouped_kernel_epsilon=_require_number(
            "grouped_kernel_epsilon", raw["grouped_kernel_epsilon"]
        ),
        grouped_workspace_budget_bytes=_require_int(
            "grouped_workspace_budget_bytes", raw["grouped_workspace_budget_bytes"]
        ),
        lora_learning_rate=_require_number(
            "lora_learning_rate", raw["lora_learning_rate"]
        ),
        lora_weight_decay=_require_number(
            "lora_weight_decay", raw["lora_weight_decay"]
        ),
        lora_beta1=_require_number("lora_beta1", raw["lora_beta1"]),
        lora_beta2=_require_number("lora_beta2", raw["lora_beta2"]),
        lora_epsilon=_require_number("lora_epsilon", raw["lora_epsilon"]),
        kernel_learning_rate=_require_number(
            "kernel_learning_rate", raw["kernel_learning_rate"]
        ),
        kernel_weight_decay=_require_number(
            "kernel_weight_decay", raw["kernel_weight_decay"]
        ),
        kernel_beta1=_require_number("kernel_beta1", raw["kernel_beta1"]),
        kernel_beta2=_require_number("kernel_beta2", raw["kernel_beta2"]),
        kernel_epsilon=_require_number("kernel_epsilon", raw["kernel_epsilon"]),
        optimizer_candidate_configurations=_require_int(
            "optimizer_candidate_configurations",
            raw["optimizer_candidate_configurations"],
        ),
        optimizer_tuning_seed_count=_require_int(
            "optimizer_tuning_seed_count", raw["optimizer_tuning_seed_count"]
        ),
        pre_registered_master_seeds=_require_seed_schedule(
            raw["pre_registered_master_seeds"]
        ),
        warmup_ratio=_require_number("warmup_ratio", raw["warmup_ratio"]),
        gradient_clip=_require_number("gradient_clip", raw["gradient_clip"]),
        gradient_accumulation_steps=_require_int(
            "gradient_accumulation_steps", raw["gradient_accumulation_steps"]
        ),
        dtype=_parse_dtype(raw["dtype"]),
        lora=LoRAConfig(
            rank=_require_int("lora.rank", lora_raw["rank"]),
            alpha=_require_number("lora.alpha", lora_raw["alpha"]),
            dropout=_require_number("lora.dropout", lora_raw["dropout"]),
            targets=_require_targets(lora_raw["targets"]),
        ),
    )
    config.validate()
    persisted_derivations = validate_persisted_seed_derivations(
        raw["seed_derivations"], master_seed=config.seed
    )
    if "replaced_layer_ids_zero_based" in raw:
        _validate_replaced_layer_ids(
            raw["replaced_layer_ids_zero_based"], expected=config.replacement_layer_ids
        )
    _validate_effective_derived_fields(raw, config)
    return config


def _normalize_historical_effective_config(
    raw: Mapping[str, object],
) -> Mapping[str, object]:
    """Supply the unused query-stat path for effective configs predating that method."""
    if not _EFFECTIVE_CONFIG_DERIVED_FIELDS.issubset(raw):
        return raw
    method = raw.get("method")
    spec = METHOD_REGISTRY.get(method) if isinstance(method, str) else None
    runs_root = raw.get("runs_root")
    if spec is None or None or (not isinstance(runs_root, str)) or (not runs_root):
        return raw
    return {**raw}


def _reject_unknown_and_missing(
    name: str,
    raw: Mapping[str, object],
    allowed_fields: frozenset[str],
    *,
    optional_fields: frozenset[str] = frozenset(),
) -> None:
    unknown = sorted(set(raw) - allowed_fields - optional_fields)
    if unknown:
        raise ValueError(f"{name} has unknown field(s): {', '.join(unknown)}")
    missing = sorted(allowed_fields - optional_fields - set(raw))
    if missing:
        raise ValueError(f"{name} is missing required field(s): {', '.join(missing)}")


def _require_string(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _require_nonempty_string(name: str, value: object) -> str:
    string_value = _require_string(name, value)
    if not string_value:
        raise ValueError(f"{name} must be a non-empty string")
    return string_value


def _parse_method(value: object) -> MethodName:
    method = _require_string("method", value)
    if method == "grouped_quadratic":
        return "grouped_quadratic"
    raise ValueError(f"method must be one of {METHOD_NAMES}; got {method!r}")


def _parse_model_profile(value: object) -> ModelProfile:
    profile = _require_string("model_profile", value)
    geometry_for_profile(profile)
    return cast(ModelProfile, profile)


def _parse_tuning_mode(value: object) -> TuningMode:
    tuning_mode = _require_string("tuning_mode", value)
    if tuning_mode == "lora":
        return "lora"
    raise ValueError(f"tuning_mode must be 'lora'; got {tuning_mode!r}")


def _parse_attention_backend(value: object) -> AttentionBackend:
    backend = _require_string("attention_backend", value)
    if backend == "legacy":
        return "legacy"
    if backend == "hd_block_gemm":
        return "hd_block_gemm"
    raise ValueError("attention_backend must be legacy or hd_block_gemm")


def _parse_hd_options(value: object) -> HDOptions | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("hd_options must be an object")
    unknown = sorted(set(value) - _HD_OPTION_FIELDS)
    if unknown:
        raise ValueError(f"hd_options has unknown field(s): {', '.join(unknown)}")
    defaults = HDOptions()
    parsed: dict[str, object] = {}
    integer_fields = {
        "token_block",
        "feature_wave_blocks",
        "memory_budget_bytes",
        "query_feature_token_tile",
        "query_fold_token_tile",
    }
    boolean_fields = {"save_local_score"}
    nullable_string_fields = (
        _HD_OPTION_FIELDS - integer_fields - boolean_fields - {"precision"}
    )
    for name in _HD_OPTION_FIELDS:
        raw_value = value.get(name, getattr(defaults, name))
        if name == "precision":
            if raw_value not in {"fp32_ieee", "bf16_tensorcore"}:
                raise ValueError(
                    "hd_options.precision must be fp32_ieee or bf16_tensorcore"
                )
        elif name in integer_fields:
            if raw_value is not None and type(raw_value) is not int:
                raise ValueError(f"hd_options.{name} must be an integer or null")
        elif name in boolean_fields:
            if raw_value is not None and type(raw_value) is not bool:
                raise ValueError(f"hd_options.{name} must be a boolean or null")
        elif name in nullable_string_fields:
            if raw_value is not None and (not isinstance(raw_value, str)):
                raise ValueError(f"hd_options.{name} must be a string or null")
        parsed[name] = raw_value
    options = HDOptions(**parsed)
    options.validate()
    return options


def _parse_group_asset_mode(value: object) -> GroupedAssetMode:
    mode = _require_string("group_asset_mode", value)
    if mode not in GROUPED_ASSET_MODES:
        raise ValueError(
            f"group_asset_mode must be one of {GROUPED_ASSET_MODES}; got {mode!r}"
        )
    return mode


def _parse_dtype(value: object) -> DTypeName:
    dtype = _require_string("dtype", value)
    if dtype != "bfloat16":
        raise ValueError(f"dtype must be 'bfloat16'; got {dtype!r}")
    return "bfloat16"


def _parse_replacement_layers(
    value: object, num_layers: int = MODEL_NUM_LAYERS, *, allow_empty: bool = False
) -> tuple[int, ...]:
    """Parse a fail-closed zero-based inclusive range or explicit ID list."""
    if isinstance(value, list):
        layer_ids = tuple(
            (_require_int("replacement_layers", layer_id) for layer_id in value)
        )
        _validate_replacement_layers(layer_ids, num_layers, allow_empty=allow_empty)
        return layer_ids
    spec = _require_string("replacement_layers", value)
    if not spec:
        if allow_empty:
            return ()
        raise ValueError("replacement_layers must not be empty")
    if ":" in spec and "," in spec:
        raise ValueError("replacement_layers cannot mix range and explicit-list forms")
    if ":" in spec:
        tokens = spec.split(":")
        if len(tokens) != 2:
            raise ValueError("replacement_layers range must have the form start:end")
        start, end = (_parse_layer_id(token) for token in tokens)
        if start > end:
            raise ValueError("replacement_layers range start must not exceed range end")
        layer_ids = tuple(range(start, end + 1))
    else:
        layer_ids = tuple((_parse_layer_id(token) for token in spec.split(",")))
    _validate_replacement_layers(layer_ids, num_layers, allow_empty=allow_empty)
    return layer_ids


def _parse_layer_id(token: str) -> int:
    if not token or not token.isascii() or (not token.isdecimal()):
        raise ValueError(
            "replacement_layers must contain nonnegative decimal layer IDs"
        )
    return int(token)


def _validate_replacement_layers(
    layer_ids: object, num_layers: int = MODEL_NUM_LAYERS, *, allow_empty: bool = False
) -> None:
    if (
        isinstance(num_layers, bool)
        or not isinstance(num_layers, int)
        or num_layers <= 0
    ):
        raise ValueError("num_layers must be a positive integer")
    if not isinstance(layer_ids, tuple) or (not layer_ids and (not allow_empty)):
        raise ValueError("replacement_layers must be a non-empty tuple of layer IDs")
    if any(
        (
            isinstance(layer_id, bool) or not isinstance(layer_id, int)
            for layer_id in layer_ids
        )
    ):
        raise ValueError("replacement_layers must contain integer layer IDs")
    if any((layer_id < 0 or layer_id >= num_layers for layer_id in layer_ids)):
        raise ValueError(
            f"replacement_layers must be in zero-based range [0, {num_layers})"
        )
    if any((left >= right for (left, right) in zip(layer_ids, layer_ids[1:]))):
        raise ValueError(
            "replacement_layers must be strictly increasing without duplicates"
        )


def _validate_replaced_layer_ids(value: object, *, expected: tuple[int, ...]) -> None:
    if not isinstance(value, list):
        raise ValueError("replaced_layer_ids_zero_based must be an array of layer IDs")
    actual = tuple(
        (_require_int("replaced_layer_ids_zero_based", layer_id) for layer_id in value)
    )
    _require_exact("replaced_layer_ids_zero_based", actual, expected)


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _require_sha256(name: str, value: object) -> str:
    digest = _require_string(name, value)
    if len(digest) != 64 or any(
        (character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 hex digest")
    return digest


def _require_seed_schedule(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or any((type(seed) is not int for seed in value)):
        raise ValueError(
            "pre_registered_master_seeds must be an array of integer seeds"
        )
    return tuple(value)


def _require_targets(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        (isinstance(item, str) for item in value)
    ):
        raise ValueError("lora.targets must be an array of strings")
    return tuple(value)


def _require_path(name: str, value: object) -> None:
    if not isinstance(value, Path) or not str(value):
        raise ValueError(f"{name} must be a non-empty path")


def _require_group_asset_path(value: object) -> None:
    if not isinstance(value, Path) or not str(value):
        raise ValueError("group_asset_path must be a non-empty path")
    group_asset_path: Path = value
    if group_asset_path == Path("."):
        raise ValueError("group_asset_path must name an asset file, not '.'")
    if group_asset_path.exists() and group_asset_path.is_dir():
        raise ValueError("group_asset_path must name an asset file, not a directory")


def _require_exact(name: str, value: object, expected: object) -> None:
    if value != expected:
        raise ValueError(f"{name} must be {expected!r}; got {value!r}")


def _require_positive_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0")


def _validate_effective_derived_fields(
    raw: Mapping[str, object], config: PilotConfig
) -> None:
    """Reject any persisted identity field that disagrees with its source config."""
    expected: Mapping[str, object] = {
        "trainable_kernel_layer_ids": list(config.trainable_kernel_layer_ids),
        "expected_group_asset_status": config.expected_group_asset_status,
        "group_asset_sequence_length": config.group_asset_sequence_length,
        "method_identity": config.method_identity,
    }
    for name, value in expected.items():
        if name in raw and raw[name] != value:
            raise ValueError(f"{name} does not match the effective PilotConfig")
