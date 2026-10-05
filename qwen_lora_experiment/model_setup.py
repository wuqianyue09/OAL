"""Model-profile-aware assembly of one pilot tuning runtime.

This module owns the only permitted construction order for the pilot.  It is
safe to import on a CPU-only development machine: Transformers is imported
only by the explicit local-loader function, and tests can inject a structural
fake model without loading weights or data.
"""

from __future__ import annotations
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
import importlib
import inspect
import math
from pathlib import Path
from typing import Literal
import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from .attention.common import (
    AttentionContractError,
    AttentionFamilyContract,
    QwenAttentionAdapter,
    Qwen2AttentionContract,
    attention_validation_checkpoint_context_fn,
)
from .attention.grouped import GroupedQuadraticQwenAttentionAdapter
from .attention.hd_runtime import HDAttentionRuntime, initialize_hd_runtime
from .backbones import llama as llama_backbone
from .backbones import qwen2 as qwen2_backbone
from .backbones.contracts import ValidatedBackbone, require_sdpa_selected
from .backbones.spec import ModelGeometry
from .config import GROUPED_BASE_METHODS, PilotConfig
from .kernel_parameters import KernelParameterBank
from .grouped_initialization import GroupedParameterInitialization
from .lora import (
    TrainableParameterReport,
    assert_all_on_device,
    assert_parameter_ownership,
    inject_lora_adapters,
)
from .package_resources import qwen2_attention_compatibility_fixture_path
from .protocol import scoped_global_rng
from .training_measurement.runtime import CheckpointMeasurementContext
from .tuning import TrainingRole, TuningPolicy, resolve_tuning_policy

KERNEL_BANK_MODULE_NAME = "_qwen_lora_kernel_bank"
_DEVICE_AUDIT_ATTRIBUTE = "_qwen_lora_device_audit"
_SETUP_ORDER = (
    "loaded",
    "validated",
    "cache_checkpointing",
    "frozen",
    "attention_replaced",
    "lora_injected",
    "kernel_registered",
    "moved",
    "audited",
    "optimizer_created",
)


class ModelSetupError(RuntimeError):
    """Raised when pilot assembly order, ownership, or device state is invalid."""


class UnsupportedQwenAttentionSignature(ModelSetupError):
    """Raised before mutation when loaded Qwen differs from the fixture."""


@dataclass(frozen=True)
class PilotOptimizer:
    """Owner-separated AdamW groups and their exact warmup-cosine schedule."""

    optimizer: Optimizer
    scheduler: LambdaLR
    warmup_steps: int
    total_steps: int


@dataclass(frozen=True)
class PilotModelRuntime:
    """All assembled objects plus evidence of their construction order."""

    model: nn.Module
    tokenizer: object
    kernel_bank: KernelParameterBank
    optimizer: Optimizer | None
    scheduler: LambdaLR | None
    tuning_policy: TuningPolicy
    training_role: TrainingRole
    trainable_report: TrainableParameterReport
    trainable_parameter_names: tuple[str, ...]
    kernel_parameter_names: tuple[str, ...]
    lora_target_names: tuple[str, ...]
    checkpointing_branch: str
    attention_replaced: bool
    attention_execution: dict[str, object]
    transformers_version: str
    device: torch.device
    setup_order: tuple[str, ...]
    warmup_steps: int


def build_model_bundle(
    config: PilotConfig,
    *,
    device: torch.device | str,
    hd_manifest_path: str | Path | None = None,
    hd_probe_path: str | Path | None = None,
    _measurement_checkpoint_context: CheckpointMeasurementContext | None = None,
) -> PilotModelRuntime:
    """Build the production bundle with the repository's pinned fixture.

    This is the stable entry point for CLIs and orchestration.  It deliberately
    exposes no model, callable, or compatibility seams: production always
    local-loads Qwen and validates the checked-in server evidence first.
    """
    kwargs: dict[str, object] = {
        "device": device,
        "compatibility_path": default_compatibility_fixture_path(),
        "hd_manifest_path": hd_manifest_path,
        "hd_probe_path": hd_probe_path,
    }
    if _measurement_checkpoint_context is not None:
        kwargs["_measurement_checkpoint_context"] = _measurement_checkpoint_context
    return assemble_lora_pilot_model(config, **kwargs)


def default_compatibility_fixture_path() -> Path:
    """Return the immutable compatibility evidence shipped inside this package."""
    fixture_path = qwen2_attention_compatibility_fixture_path()
    if not fixture_path.is_file():
        raise ModelSetupError(
            "installed package is missing qwen_lora_experiment/fixtures/qwen2_attention_compatibility.json"
        )
    return fixture_path


def load_local_qwen_model_and_tokenizer(
    config: PilotConfig, *, importer: Callable[[str], object] = importlib.import_module
) -> tuple[nn.Module, object]:
    """Preserve the historical Qwen-named entry over profile dispatch."""
    return load_local_model_and_tokenizer(config, importer=importer)


def load_local_model_and_tokenizer(
    config: PilotConfig, *, importer: Callable[[str], object] = importlib.import_module
) -> tuple[nn.Module, object]:
    """Load one supported local base and its tokenizer from the same path."""
    config.validate()
    try:
        transformers = importer("transformers")
        model_loader = getattr(
            getattr(transformers, "AutoModelForCausalLM"), "from_pretrained"
        )
        tokenizer_loader = getattr(
            getattr(transformers, "AutoTokenizer"), "from_pretrained"
        )
    except Exception as exc:
        raise ModelSetupError(
            "could not access Transformers local model loaders"
        ) from exc
    if not callable(model_loader) or not callable(tokenizer_loader):
        raise ModelSetupError("Transformers local model loaders are not callable")
    model_path = str(config.model_path)
    try:
        model = model_loader(
            model_path,
            local_files_only=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        )
        tokenizer = tokenizer_loader(model_path, local_files_only=True)
    except Exception as exc:
        raise ModelSetupError(
            f"local {config.resolved_model_profile} model/tokenizer load failed with local_files_only=True"
        ) from exc
    if not isinstance(model, nn.Module):
        raise ModelSetupError(
            "AutoModelForCausalLM.from_pretrained must return nn.Module"
        )
    return (model, tokenizer)


def assemble_lora_pilot_model(
    config: PilotConfig,
    *,
    model: nn.Module | None = None,
    tokenizer: object | None = None,
    device: torch.device | str,
    compatibility_path: str | Path,
    apply_rotary_pos_emb: (
        Callable[[Tensor, Tensor, Tensor, Tensor], object] | None
    ) = None,
    hd_manifest_path: str | Path | None = None,
    hd_probe_path: str | Path | None = None,
    importer: Callable[[str], object] = importlib.import_module,
    nonformal_test_mode: bool = False,
    _measurement_checkpoint_context: CheckpointMeasurementContext | None = None,
) -> PilotModelRuntime:
    """Assemble one pilot without leaking construction RNG mutations."""
    config.validate()
    if type(nonformal_test_mode) is not bool:
        raise TypeError("nonformal_test_mode must be a boolean")
    injected_construction_seam = (
        any(value is not None for value in (model, tokenizer, apply_rotary_pos_emb))
        or importer is not importlib.import_module
    )
    if injected_construction_seam and (not nonformal_test_mode):
        raise ModelSetupError(
            "injected model/callable construction seams require nonformal_test_mode=True"
        )
    with scoped_global_rng(config.seed_derivations.training_global_rng_seed):
        return _assemble_lora_pilot_model_seeded(
            config,
            model=model,
            tokenizer=tokenizer,
            device=device,
            compatibility_path=compatibility_path,
            apply_rotary_pos_emb=apply_rotary_pos_emb,
            hd_manifest_path=hd_manifest_path,
            hd_probe_path=hd_probe_path,
            importer=importer,
            nonformal_test_mode=nonformal_test_mode,
            _measurement_checkpoint_context=_measurement_checkpoint_context,
        )


def _assemble_lora_pilot_model_seeded(
    config: PilotConfig,
    *,
    model: nn.Module | None = None,
    tokenizer: object | None = None,
    device: torch.device | str,
    compatibility_path: str | Path,
    apply_rotary_pos_emb: (
        Callable[[Tensor, Tensor, Tensor, Tensor], object] | None
    ) = None,
    hd_manifest_path: str | Path | None = None,
    hd_probe_path: str | Path | None = None,
    importer: Callable[[str], object] = importlib.import_module,
    nonformal_test_mode: bool = False,
    _measurement_checkpoint_context: CheckpointMeasurementContext | None = None,
) -> PilotModelRuntime:
    """Assemble exactly one method/mode pilot in its required order.

    A caller supplying ``model`` and ``tokenizer`` is used only by CPU tests
    and checkpoint restoration.  Supplying either one without the other is
    rejected; production normal operation always enters through the strict
    local loader above.
    """
    config.validate()
    tuning_policy = resolve_tuning_policy(config.tuning_mode, config.method)
    target_device = _require_device(device)
    hd_runtime = initialize_hd_runtime(
        config,
        target_device,
        manifest_path=hd_manifest_path,
        runtime_probe_path=hd_probe_path,
        importer=importer,
    )
    trace: list[str] = []
    if (model is None) != (tokenizer is None):
        raise ModelSetupError(
            "model and tokenizer must be supplied together or both loaded locally"
        )
    from .precomputed_assets import load_precomputed_group_asset

    precomputed_initialization = load_precomputed_group_asset(config).initialization
    if model is None:
        model, tokenizer = load_local_qwen_model_and_tokenizer(
            config, importer=importer
        )
    assert tokenizer is not None
    trace.append("loaded")
    validated_backbone = _validate_loaded_model(
        model,
        model_profile=config.resolved_model_profile,
        compatibility_path=compatibility_path,
        importer=importer,
        supplied_apply_rotary_pos_emb=apply_rotary_pos_emb,
    )
    contract = validated_backbone.compatibility_contract
    runtime_rotary = validated_backbone.apply_rotary_pos_emb
    trace.append("validated")
    checkpointing_branch = _configure_no_cache_checkpointing(
        model, checkpoint_measurement_context=_measurement_checkpoint_context
    )
    trace.append("cache_checkpointing")
    _freeze_all_parameters(model)
    trace.append("frozen")
    kernel_bank = _build_kernel_bank(
        config, grouped_initialization=precomputed_initialization
    )
    replacement_layer_ids = config.replacement_layer_ids
    attention_replaced = bool(replacement_layer_ids)
    if attention_replaced:
        _replace_attention_modules(
            model,
            contract=contract,
            method=config.method,
            replacement_layer_ids=replacement_layer_ids,
            kernel_bank=kernel_bank,
            compatibility_path=compatibility_path,
            apply_rotary_pos_emb=runtime_rotary,
            hd_runtime=hd_runtime,
            geometry=validated_backbone.geometry,
            family_contract=validated_backbone.family_contract,
        )
        trace.append("attention_replaced")
    else:
        _require_sdpa_selected(model)
        trace.append("attention_preserved_sdpa")
    lora_targets = tuple(
        inject_lora_adapters(
            model,
            config.lora,
            seed=config.seed_derivations.lora_init_seed,
            geometry=validated_backbone.geometry,
        )
    )
    trace.append("lora_injected")
    _register_kernel_bank_once(model, kernel_bank)
    trace.append("kernel_registered")
    model.to(device=target_device, dtype=torch.bfloat16)
    trace.append("moved")
    kernel_parameter_names = _kernel_parameter_names(config.method, kernel_bank)
    trainable_report = _audit_runtime_ownership_and_device(
        model,
        method=config.method,
        device=target_device,
        kernel_parameter_names=kernel_parameter_names,
        tuning_policy=tuning_policy,
        geometry=validated_backbone.geometry,
    )
    trace.append("audited")
    optimizer_bundle: PilotOptimizer | None
    optimizer_bundle = _build_pilot_optimizer_for_policy(
        config, model, target_device, tuning_policy=tuning_policy
    )
    trace.append("optimizer_created")
    expected_order = list(_SETUP_ORDER)
    if not attention_replaced:
        expected_order[4] = "attention_preserved_sdpa"
    if tuple(trace) != tuple(expected_order):
        raise ModelSetupError(
            "internal pilot setup order does not match the approved contract"
        )
    attention_execution = _actual_attention_execution_evidence(
        config, model, attention_replaced=attention_replaced
    )
    return PilotModelRuntime(
        model=model,
        tokenizer=tokenizer,
        kernel_bank=kernel_bank,
        optimizer=optimizer_bundle.optimizer if optimizer_bundle is not None else None,
        scheduler=optimizer_bundle.scheduler if optimizer_bundle is not None else None,
        tuning_policy=tuning_policy,
        training_role=tuning_policy.training_role,
        trainable_report=trainable_report,
        trainable_parameter_names=trainable_report.trainable_names,
        kernel_parameter_names=kernel_parameter_names,
        lora_target_names=lora_targets,
        checkpointing_branch=checkpointing_branch,
        attention_replaced=attention_replaced,
        attention_execution=attention_execution,
        transformers_version=validated_backbone.transformers_version,
        device=target_device,
        setup_order=tuple(trace),
        warmup_steps=(
            optimizer_bundle.warmup_steps if optimizer_bundle is not None else 0
        ),
    )


def _attention_adapter_type(method: str) -> type[QwenAttentionAdapter] | None:
    """Resolve the adapter used by both replacement and execution evidence."""
    return {"grouped_quadratic": GroupedQuadraticQwenAttentionAdapter}.get(method)


def _actual_attention_execution_evidence(
    config: PilotConfig, model: nn.Module, *, attention_replaced: bool
) -> dict[str, object]:
    """Record the callable after proving exactly the selected layers use it."""
    layers = getattr(getattr(model, "model", None), "layers", None)
    geometry = config.geometry
    if not isinstance(layers, nn.ModuleList) or len(layers) != geometry.num_layers:
        raise ModelSetupError(
            f"cannot record execution without all configured {geometry.num_layers} {config.resolved_model_profile} attention layers"
        )
    expected_type = _attention_adapter_type(config.method)
    expected_layer_ids = config.replacement_layer_ids
    actual_adapter_layer_ids = tuple(
        (
            layer_id
            for (layer_id, layer) in enumerate(layers)
            if isinstance(layer.self_attn, QwenAttentionAdapter)
        )
    )
    if (
        not attention_replaced
        or expected_type is None
        or actual_adapter_layer_ids != expected_layer_ids
        or any(
            (
                not isinstance(layers[layer_id].self_attn, expected_type)
                for layer_id in expected_layer_ids
            )
        )
    ):
        raise ModelSetupError(
            f"{config.method} execution evidence does not match the selected attention layers"
        )
    if config.attention_backend == "hd_block_gemm":
        adapters = tuple(
            (layers[layer_id].self_attn for layer_id in expected_layer_ids)
        )
        runtime = getattr(adapters[0], "_hd_runtime", None)
        if not isinstance(runtime, HDAttentionRuntime) or any(
            (
                getattr(adapter, "_hd_runtime", None) is not runtime
                for adapter in adapters
            )
        ):
            raise ModelSetupError(
                "HD execution evidence requires one shared initialized runtime"
            )
        family_callables = {
            "grouped_quadratic": "oal_attention.hd_block_gemm_adapters.run_prepared_grouped_hd_block_gemm"
        }
        return {
            "method": config.method,
            "method_identity": config.method_identity,
            "execution": config.attention_execution,
            "callable": family_callables[config.method],
            "experimental": True,
            "precision": runtime.options["precision"],
            "requested_options": runtime.requested_execution_options(),
            "options": runtime.execution_options(),
            "logical_feature_width": 2145,
            "packed_pair_count": 2080,
            "physical_repeat_kv": False,
            "replaced_layer_ids_zero_based": list(expected_layer_ids),
        }
    return {
        "method": "grouped_quadratic",
        "method_identity": config.method_identity,
        "execution": "triton",
        "callable": "oal_attention.oal_attention",
        "experimental": False,
        "admission": "public_fail_closed",
        "group_asset_mode": config.group_asset_mode,
        "group_asset_status": config.expected_group_asset_status,
        "group_asset_sequence_length": config.group_asset_sequence_length,
        "replaced_layer_ids_zero_based": list(expected_layer_ids),
    }


def build_pilot_optimizer(
    config: PilotConfig, model: nn.Module, device: torch.device | str
) -> PilotOptimizer:
    """Create separate LoRA and OAL optimizer groups after the device audit."""
    config.validate()
    tuning_policy = resolve_tuning_policy(config.tuning_mode, config.method)
    return _build_pilot_optimizer_for_policy(
        config, model, device, tuning_policy=tuning_policy
    )


def _build_pilot_optimizer_for_policy(
    config: PilotConfig,
    model: nn.Module,
    device: torch.device | str,
    *,
    tuning_policy: TuningPolicy,
) -> PilotOptimizer:
    """Build optimizer groups from the already-resolved assembly policy."""
    target_device = _require_device(device)
    audited_device = getattr(model, _DEVICE_AUDIT_ATTRIBUTE, None)
    if audited_device != target_device:
        raise ModelSetupError(
            "optimizer creation requires the completed post-migration device audit"
        )
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise ModelSetupError(
            "optimizer creation found no audited trainable parameters"
        )
    if any((parameter.device != target_device for parameter in trainable)):
        raise ModelSetupError(
            "optimizer creation found a trainable parameter on the wrong device"
        )
    kernel_bank = model.get_submodule(KERNEL_BANK_MODULE_NAME)
    if not isinstance(kernel_bank, KernelParameterBank):
        raise ModelSetupError(
            "optimizer creation cannot find the registered kernel bank"
        )
    kernel_parameter_ids = {
        id(parameter)
        for parameter in kernel_bank.parameters()
        if parameter.requires_grad
    }
    lora_parameters: list[nn.Parameter] = []
    kernel_parameters: list[nn.Parameter] = []
    for parameter in trainable:
        if id(parameter) in kernel_parameter_ids:
            kernel_parameters.append(parameter)
        else:
            lora_parameters.append(parameter)
    if {id(parameter) for parameter in kernel_parameters} != kernel_parameter_ids:
        raise ModelSetupError(
            "optimizer creation kernel ownership does not match the active method-specific scope"
        )
    groups: list[dict[str, object]] = []
    if not lora_parameters:
        raise ModelSetupError("optimizer creation requires LoRA trainable parameters")
    groups.append(
        {
            "params": lora_parameters,
            "lr": config.lora_learning_rate,
            "weight_decay": config.lora_weight_decay,
            "betas": (config.lora_beta1, config.lora_beta2),
            "eps": config.lora_epsilon,
            "owner": "lora",
        }
    )
    if kernel_parameters:
        groups.append(
            {
                "params": kernel_parameters,
                "lr": config.kernel_learning_rate,
                "weight_decay": config.kernel_weight_decay,
                "betas": (config.kernel_beta1, config.kernel_beta2),
                "eps": config.kernel_epsilon,
                "owner": "kernel",
            }
        )
    optimizer = AdamW(groups)
    optimized = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    if len(optimized) != len({id(parameter) for parameter in optimized}):
        raise ModelSetupError("pilot optimizer contains duplicate parameter ownership")
    if {id(parameter) for parameter in optimized} != {
        id(parameter) for parameter in trainable
    }:
        raise ModelSetupError(
            "pilot optimizer parameters do not exactly match audited trainables"
        )
    expected_group_count = 1 + int(bool(kernel_parameters))
    if len(optimizer.param_groups) != expected_group_count:
        raise ModelSetupError(
            "pilot optimizer group count does not match active ownership"
        )
    total_steps = config.train_blocks
    warmup_steps = max(1, math.ceil(total_steps * config.warmup_ratio))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max(1, total_steps - warmup_steps)
        progress = min(1.0, float(step - warmup_steps) / float(remaining))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=schedule)
    return PilotOptimizer(
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )


def _validate_loaded_model(
    model: object,
    *,
    model_profile: str,
    compatibility_path: str | Path,
    importer: Callable[[str], object],
    supplied_apply_rotary_pos_emb: (
        Callable[[Tensor, Tensor, Tensor, Tensor], object] | None
    ),
) -> ValidatedBackbone:
    """Dispatch model validation without applying Qwen's fixture to Llama."""
    try:
        if model_profile == qwen2_backbone.PROFILE:
            validated = qwen2_backbone.validate_model(
                model,
                compatibility_path=compatibility_path,
                importer=importer,
                supplied_apply_rotary_pos_emb=supplied_apply_rotary_pos_emb,
            )
        elif model_profile == llama_backbone.PROFILE:
            validated = llama_backbone.validate_model(
                model,
                importer=importer,
                supplied_apply_rotary_pos_emb=supplied_apply_rotary_pos_emb,
            )
        else:
            raise UnsupportedQwenAttentionSignature(
                f"unsupported model profile: {model_profile!r}"
            )
    except (AttentionContractError, ValueError) as exc:
        raise UnsupportedQwenAttentionSignature(str(exc)) from exc
    return validated


def _require_sdpa_selected(model: nn.Module) -> None:
    """Translate a post-assembly backend failure into the public setup error."""
    try:
        require_sdpa_selected(model)
    except AttentionContractError as exc:
        raise UnsupportedQwenAttentionSignature(str(exc)) from exc


def _configure_no_cache_checkpointing(
    model: nn.Module,
    *,
    checkpoint_measurement_context: CheckpointMeasurementContext | None = None,
) -> Literal["nonreentrant", "compat_enable_input_require_grads"]:
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "use_cache"):
        raise ModelSetupError("loaded Qwen config.use_cache is required")
    config.use_cache = False
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and hasattr(generation_config, "use_cache"):
        generation_config.use_cache = False
    checkpointing_enable = getattr(model, "gradient_checkpointing_enable", None)
    if not callable(checkpointing_enable):
        raise ModelSetupError("loaded Qwen model lacks gradient_checkpointing_enable")
    if checkpoint_measurement_context is not None:
        return _configure_measured_checkpointing(
            checkpointing_enable, checkpoint_measurement_context
        )
    accepts_checkpointing_kwargs = _checkpointing_kwargs_support(checkpointing_enable)
    if accepts_checkpointing_kwargs is False:
        return _configure_legacy_checkpointing(model, checkpointing_enable)
    gradient_checkpointing_kwargs: dict[str, object] = {"use_reentrant": False}
    if _torch_checkpoint_context_fn_support():
        gradient_checkpointing_kwargs["context_fn"] = (
            attention_validation_checkpoint_context_fn
        )
    try:
        checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )
    except TypeError as initial_error:
        if (
            accepts_checkpointing_kwargs is None
            and _is_unexpected_checkpointing_keyword_error(initial_error)
        ):
            return _configure_legacy_checkpointing(model, checkpointing_enable)
        raise ModelSetupError(
            "nonreentrant Qwen gradient-checkpointing setup raised TypeError"
        ) from initial_error
    except Exception as exc:
        raise ModelSetupError(
            "nonreentrant Qwen gradient-checkpointing setup failed"
        ) from exc
    return "nonreentrant"


def _configure_measured_checkpointing(
    checkpointing_enable: Callable[..., object],
    checkpoint_measurement_context: CheckpointMeasurementContext,
) -> Literal["nonreentrant"]:
    """Enable only context-aware non-reentrant checkpointing for training measurement.

    A replay outside the original operator trace session cannot be attributed to
    its optimizer-step window.  Therefore training measurement rejects both the legacy
    branch and any runtime that refuses ``context_fn`` rather than silently
    changing checkpoint behavior.
    """
    context_fn = getattr(checkpoint_measurement_context, "checkpoint_context_fn", None)
    if not callable(context_fn):
        raise ModelSetupError(
            "training measurement checkpoint preflight requires a callable checkpoint_context_fn"
        )
    if not _torch_checkpoint_context_fn_support():
        raise ModelSetupError(
            "training measurement checkpoint preflight requires PyTorch checkpoint context_fn support"
        )
    if _checkpointing_kwargs_support(checkpointing_enable) is not True:
        raise ModelSetupError(
            "training measurement checkpoint preflight requires Transformers nonreentrant checkpoint kwargs"
        )
    composed_context_fn = _compose_checkpoint_context_fns(context_fn)
    try:
        checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
                "context_fn": composed_context_fn,
            }
        )
    except Exception as exc:
        raise ModelSetupError(
            "training measurement checkpoint preflight rejected nonreentrant context_fn"
        ) from exc
    return "nonreentrant"


@contextmanager
def _nested_checkpoint_contexts(
    outer: AbstractContextManager[None], inner: AbstractContextManager[None]
) -> Iterator[None]:
    """Enter a profiling context outside its attention-validation context."""
    with outer:
        with inner:
            yield


def _compose_checkpoint_context_fns(
    measurement_context_fn: Callable[
        [], tuple[AbstractContextManager[None], AbstractContextManager[None]]
    ],
) -> Callable[[], tuple[AbstractContextManager[None], AbstractContextManager[None]]]:
    """Compose training measurement profiling with per-frame validation replay state."""

    def composed_context_fn() -> (
        tuple[AbstractContextManager[None], AbstractContextManager[None]]
    ):
        measurement_original, measurement_recompute = measurement_context_fn()
        validation_original, validation_recompute = (
            attention_validation_checkpoint_context_fn()
        )
        return (
            _nested_checkpoint_contexts(measurement_original, validation_original),
            _nested_checkpoint_contexts(measurement_recompute, validation_recompute),
        )

    return composed_context_fn


def _torch_checkpoint_context_fn_support() -> bool:
    """Fail closed if this installed PyTorch cannot carry checkpoint contexts."""
    try:
        from torch.utils.checkpoint import checkpoint

        parameters = inspect.signature(checkpoint).parameters.values()
    except (ImportError, TypeError, ValueError):
        return False
    return any(
        (
            parameter.name == "context_fn"
            and parameter.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            for parameter in parameters
        )
    )


def _checkpointing_kwargs_support(
    checkpointing_enable: Callable[..., object],
) -> bool | None:
    """Return whether the callable's inspectable signature accepts the keyword."""
    try:
        parameters = inspect.signature(checkpointing_enable).parameters.values()
    except (TypeError, ValueError):
        return None
    for parameter in parameters:
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == "gradient_checkpointing_kwargs" and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


def _is_unexpected_checkpointing_keyword_error(error: TypeError) -> bool:
    message = str(error)
    return (
        "gradient_checkpointing_kwargs" in message
        and "unexpected keyword" in message.lower()
    )


def _configure_legacy_checkpointing(
    model: nn.Module, checkpointing_enable: Callable[..., object]
) -> Literal["compat_enable_input_require_grads"]:
    enable_input_require_grads = getattr(model, "enable_input_require_grads", None)
    if not callable(enable_input_require_grads):
        raise ModelSetupError(
            "legacy Qwen checkpointing requires enable_input_require_grads"
        )
    try:
        enable_input_require_grads()
        checkpointing_enable()
    except Exception as exc:
        raise ModelSetupError(
            "compatible Qwen gradient-checkpointing setup failed"
        ) from exc
    return "compat_enable_input_require_grads"


def _freeze_all_parameters(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def _build_kernel_bank(
    config: PilotConfig,
    *,
    grouped_initialization: GroupedParameterInitialization | None = None,
) -> KernelParameterBank:
    if grouped_initialization is None:
        from .precomputed_assets import load_precomputed_group_asset

        grouped_initialization = load_precomputed_group_asset(config).initialization
    return KernelParameterBank(
        config.method,
        geometry=config.geometry,
        grouped_initialization=grouped_initialization,
        grouped_kernel_epsilon=config.grouped_kernel_epsilon,
        trainable_kernel_layer_ids=config.trainable_kernel_layer_ids,
    )


def _replace_attention_modules(
    model: nn.Module,
    *,
    contract: Qwen2AttentionContract | None,
    method: str,
    replacement_layer_ids: tuple[int, ...],
    kernel_bank: KernelParameterBank,
    compatibility_path: str | Path,
    apply_rotary_pos_emb: Callable[[Tensor, Tensor, Tensor, Tensor], object] | None,
    hd_runtime: HDAttentionRuntime | None,
    geometry: ModelGeometry,
    family_contract: AttentionFamilyContract,
) -> None:
    del contract
    layers = model.model.layers
    selected_layer_ids = frozenset(replacement_layer_ids)
    if not selected_layer_ids:
        raise ModelSetupError(
            "custom attention replacement requires at least one selected layer"
        )
    adapter_type = _attention_adapter_type(method)
    for layer_id, layer in enumerate(layers):
        if layer_id not in selected_layer_ids:
            continue
        original_attention = layer.self_attn
        common_kwargs: dict[str, object] = {
            "layer_id": layer_id,
            "compatibility_path": compatibility_path,
            "apply_rotary_pos_emb": apply_rotary_pos_emb,
            "geometry": geometry,
            "family_contract": family_contract,
        }
        if method == "grouped_quadratic":
            replacement = adapter_type(
                original_attention,
                grouped_parameter_bank=kernel_bank,
                hd_runtime=hd_runtime,
                **common_kwargs,
            )
        else:
            raise ModelSetupError(f"unsupported custom attention method: {method}")
        layer.self_attn = replacement


def _register_kernel_bank_once(model: nn.Module, bank: KernelParameterBank) -> None:
    if KERNEL_BANK_MODULE_NAME in model._modules or hasattr(
        model, KERNEL_BANK_MODULE_NAME
    ):
        raise ModelSetupError("kernel bank is already registered on the model")
    if any((module is bank for module in model.modules())):
        raise ModelSetupError(
            "kernel bank must not be registered before its unique model owner"
        )
    model.add_module(KERNEL_BANK_MODULE_NAME, bank)
    if model.get_submodule(KERNEL_BANK_MODULE_NAME) is not bank:
        raise ModelSetupError(
            "kernel bank registration did not retain the supplied bank"
        )
    if sum((module is bank for module in model.modules())) != 1:
        raise ModelSetupError("kernel bank must be registered exactly once")


def _kernel_parameter_names(method: str, bank: KernelParameterBank) -> tuple[str, ...]:
    if bank.method != method:
        raise ModelSetupError(
            "kernel bank method does not match parameter-name request"
        )
    names = tuple(
        (
            f"{KERNEL_BANK_MODULE_NAME}.{name}"
            for (name, parameter) in bank.named_parameters()
            if parameter.requires_grad
        )
    )
    return names


def _audit_runtime_ownership_and_device(
    model: nn.Module,
    *,
    method: str,
    device: torch.device,
    kernel_parameter_names: tuple[str, ...],
    tuning_policy: TuningPolicy | None = None,
    geometry: ModelGeometry | None = None,
) -> TrainableParameterReport:
    policy = tuning_policy or resolve_tuning_policy("lora", method)
    report = assert_parameter_ownership(
        model, method, kernel_parameter_names=kernel_parameter_names, geometry=geometry
    )
    assert_all_on_device(
        model, device=device, kernel_parameter_names=kernel_parameter_names
    )
    _assert_all_buffers_on_device(model, device)
    setattr(model, _DEVICE_AUDIT_ATTRIBUTE, device)
    return report


def _assert_all_buffers_on_device(model: nn.Module, device: torch.device) -> None:
    for name, buffer in model.named_buffers():
        if buffer.device != device:
            raise ModelSetupError(
                f"device audit found buffer {name} on {buffer.device}, expected {device}"
            )
    bank = model.get_submodule(KERNEL_BANK_MODULE_NAME)
    if not isinstance(bank, KernelParameterBank):
        raise ModelSetupError(
            "device audit cannot find the uniquely registered kernel bank"
        )
    if bank.method in GROUPED_BASE_METHODS:
        if bank.dim_groups.dtype is not torch.int32 or bank.dim_groups.requires_grad:
            raise ModelSetupError("Grouped dim_groups must remain frozen int32")
        if bank.grouped_kernel_epsilon.dtype is not torch.float32:
            raise ModelSetupError("Grouped kernel epsilon must remain FP32")
        if (
            bank.group_counts.dtype is not torch.int32
            or bank.group_counts.requires_grad
        ):
            raise ModelSetupError("Grouped group_counts must remain frozen int32")


def _require_device(device: torch.device | str) -> torch.device:
    try:
        resolved = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ModelSetupError("device must be a valid torch device") from exc
    if resolved.type == "meta":
        raise ModelSetupError("model setup cannot target the meta device")
    return resolved
