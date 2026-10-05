"""CPU release checks for isolated OAL imports, assembly and checkpoints."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import importlib
import inspect
import pkgutil
import pytest
import torch
from torch import Tensor, nn
from qwen_lora_experiment.config import from_json_file, from_mapping, METHOD_NAMES
from qwen_lora_experiment.model_setup import assemble_lora_pilot_model
from qwen_lora_experiment.lora import LoRAAdapter

ROOT = Path(__file__).resolve().parents[1]


class FakeQwen2Attention(nn.Module):
    """A no-forward, structural Qwen2Attention with the measured signature."""

    def __init__(self) -> None:
        super().__init__()
        self.head_dim = 64
        self.q_proj = nn.Linear(896, 896, bias=True, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(896, 128, bias=True, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(896, 128, bias=True, dtype=torch.bfloat16)
        self.o_proj = nn.Linear(896, 896, bias=False, dtype=torch.bfloat16)

    def forward(self, *args: object, **kwargs: object) -> tuple[Tensor, None]:
        del args, kwargs
        raise AssertionError("fake Qwen attention has no model-forward test path")


class FakeLayer(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.self_attn = FakeQwen2Attention()
        self.mlp = nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)


class FakeBackbone(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList((FakeLayer() for _ in range(24)))


class FakeQwenForCausalLM(nn.Module):

    def __init__(
        self,
        *,
        raise_internal_checkpoint_typeerror: bool = False,
        raise_keyword_looking_internal_typeerror: bool = False,
    ) -> None:
        super().__init__()
        self.model = FakeBackbone()
        self.lm_head = nn.Linear(896, 16, bias=False, dtype=torch.bfloat16)
        self.config = SimpleNamespace(
            model_type="qwen2",
            num_hidden_layers=24,
            num_attention_heads=14,
            num_key_value_heads=2,
            hidden_size=896,
            _attn_implementation="sdpa",
            use_cache=True,
        )
        self.generation_config = SimpleNamespace(use_cache=True)
        self.events: list[str] = []
        self.checkpointing_kwargs: list[dict[str, object]] = []
        self.raise_internal_checkpoint_typeerror = raise_internal_checkpoint_typeerror
        self.raise_keyword_looking_internal_typeerror = (
            raise_keyword_looking_internal_typeerror
        )

    def gradient_checkpointing_enable(self, **kwargs: object) -> None:
        self.events.append(f"checkpoint:{kwargs!r}")
        self.checkpointing_kwargs.append(dict(kwargs))
        if self.raise_internal_checkpoint_typeerror and kwargs:
            raise TypeError("internal checkpointing bug")
        if self.raise_keyword_looking_internal_typeerror and kwargs:
            raise TypeError(
                "helper got an unexpected keyword argument 'gradient_checkpointing_kwargs'"
            )

    def enable_input_require_grads(self) -> None:
        self.events.append("enable_input_require_grads")


class FakeTokenizer:
    pass


def _runtime_rotary(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: object | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[Tensor, Tensor]:
    del cos, sin
    del position_ids, unsqueeze_dim
    return (q, k)


def _runtime_importer(
    *,
    version: str = "4.51.3",
    attention_class: type[nn.Module] = FakeQwen2Attention,
    rotary: object = _runtime_rotary,
) -> object:
    transformers = SimpleNamespace(__version__=version)
    modeling = SimpleNamespace(
        Qwen2Attention=attention_class, apply_rotary_pos_emb=rotary
    )

    def importer(name: str) -> object:
        if name == "transformers":
            return transformers
        if name == "transformers.models.qwen2.modeling_qwen2":
            return modeling
        raise ModuleNotFoundError(name=name)

    return importer


class _RenderedAnnotation:

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return self.text


_QWEN_FORWARD_SIGNATURE = inspect.Signature(
    parameters=(
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter(
            "hidden_states",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=_RenderedAnnotation("torch.Tensor"),
        ),
        inspect.Parameter(
            "position_embeddings",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=_RenderedAnnotation("Tuple[torch.Tensor, torch.Tensor]"),
        ),
        inspect.Parameter(
            "attention_mask",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=_RenderedAnnotation("Optional[torch.Tensor]"),
        ),
        inspect.Parameter(
            "past_key_value",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=None,
            annotation=_RenderedAnnotation("Optional[transformers.cache_utils.Cache]"),
        ),
        inspect.Parameter(
            "cache_position",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=None,
            annotation=_RenderedAnnotation("Optional[torch.LongTensor]"),
        ),
        inspect.Parameter(
            "kwargs",
            inspect.Parameter.VAR_KEYWORD,
            annotation=_RenderedAnnotation(
                "typing_extensions.Unpack[transformers.modeling_flash_attention_utils.FlashAttentionKwargs]"
            ),
        ),
    ),
    return_annotation=_RenderedAnnotation(
        "Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]"
    ),
)

FakeQwen2Attention.__name__ = "Qwen2Attention"

FakeQwen2Attention.__qualname__ = "Qwen2Attention"

FakeQwen2Attention.__module__ = "transformers.models.qwen2.modeling_qwen2"

FakeQwen2Attention.forward.__signature__ = _QWEN_FORWARD_SIGNATURE

_runtime_rotary.__signature__ = inspect.Signature(
    parameters=(
        inspect.Parameter("q", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("k", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("cos", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("sin", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter(
            "position_ids", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
        ),
        inspect.Parameter(
            "unsqueeze_dim", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=1
        ),
    )
)


@pytest.fixture
def runtime():
    config = from_json_file(ROOT / "configs/lora_qwen_n4096_template.json")
    return assemble_lora_pilot_model(
        config,
        model=FakeQwenForCausalLM(),
        tokenizer=FakeTokenizer(),
        device="cpu",
        compatibility_path=ROOT
        / "qwen_lora_experiment/fixtures/qwen2_attention_compatibility.json",
        apply_rotary_pos_emb=_runtime_rotary,
        importer=_runtime_importer(),
        nonformal_test_mode=True,
    )


def test_only_oal_configuration_is_supported_and_roundtrips():
    config = from_json_file(ROOT / "configs/lora_qwen_n4096_template.json")
    assert METHOD_NAMES == ("grouped_quadratic",)
    assert from_mapping(config.to_dict()) == config
    for alternative in (
        "performer",
        "spectraformer",
        "dijiang",
        "dijiang_new",
        "taylorshift",
        "learned_global",
        "softmax",
    ):
        with pytest.raises(ValueError):
            replace(config, method=alternative).validate()
    assert config.seed_derivations.lora_init_seed == 7343342373812646833
    assert config.seed_derivations.training_global_rng_seed == 1386572687246239682
    assert config.seed_derivations.data_permutation_seed == 447239342449798306


def test_release_modules_import_without_alternative_implementations():
    import oal_attention, qwen_lora_experiment

    for package in (oal_attention, qwen_lora_experiment):
        for name in getattr(package, "__all__", ()):
            assert hasattr(package, name), (package.__name__, name)
        for info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            module = importlib.import_module(info.name)
            for name in getattr(module, "__all__", ()):
                assert hasattr(module, name), (info.name, name)


def test_oal_lora_assembly_keeps_layers_and_parameter_ownership(runtime):
    from qwen_lora_experiment.attention.grouped import (
        GroupedQuadraticQwenAttentionAdapter,
    )

    assert tuple(
        i
        for i, layer in enumerate(runtime.model.model.layers)
        if isinstance(layer.self_attn, GroupedQuadraticQwenAttentionAdapter)
    ) == tuple(range(3, 21))
    adapters = [
        module for module in runtime.model.modules() if isinstance(module, LoRAAdapter)
    ]
    assert len(adapters) == 24 * 4
    assert all(
        p.dtype == torch.float32 for module in adapters for p in module.parameters()
    )
    assert all(p.dtype == torch.float32 for p in runtime.kernel_bank.parameters())
    assert [group["owner"] for group in runtime.optimizer.param_groups] == [
        "lora",
        "kernel",
    ]
    optimized = {
        id(p) for group in runtime.optimizer.param_groups for p in group["params"]
    }
    assert optimized == {id(p) for p in runtime.model.parameters() if p.requires_grad}
    assert runtime.attention_execution["group_asset_status"] == "precomputed"
    assert tuple(runtime.kernel_bank.dim_groups.shape) == (18, 14, 64)


def test_best_checkpoint_restores_lora_and_oal_in_place(tmp_path, runtime):
    from qwen_lora_experiment.checkpointing import (
        CheckpointContext,
        save_best_adapter,
        load_best_adapter,
    )

    context = CheckpointContext(
        method="grouped_quadratic",
        config_identity={
            "checkpoint_context_kind": "nonformal_test",
            "schema": 1,
            "method": "grouped_quadratic",
            "sequence_length": 4096,
        },
        data_identity={"manifest_sha256": "test-data"},
        model_identity={"identity_sha256": "test-model"},
    )
    before = {
        name: p.detach().clone()
        for name, p in runtime.model.named_parameters()
        if p.requires_grad
    }
    identities = {
        name: id(p) for name, p in runtime.model.named_parameters() if p.requires_grad
    }
    save_best_adapter(
        tmp_path,
        model=runtime.model,
        kernel_bank=runtime.kernel_bank,
        context=context,
        step=2,
        validation_nll=1.0,
    )
    with torch.no_grad():
        for p in runtime.model.parameters():
            if p.requires_grad:
                p.add_(0.5)
    load_best_adapter(
        tmp_path, model=runtime.model, kernel_bank=runtime.kernel_bank, context=context
    )
    after = {name: p for name, p in runtime.model.named_parameters() if p.requires_grad}
    assert {name: id(p) for name, p in after.items()} == identities
    assert all(torch.equal(p, before[name]) for name, p in after.items())


def test_resume_restores_parameters_optimizer_scheduler_cursor_and_rng(
    tmp_path, runtime
):
    import copy
    import random
    import numpy as np
    from qwen_lora_experiment.checkpointing import (
        CheckpointContext,
        save_latest_resume,
        load_latest_resume,
    )
    from qwen_lora_experiment.protocol import preserve_global_rng_state

    context = CheckpointContext(
        method="grouped_quadratic",
        config_identity={
            "checkpoint_context_kind": "nonformal_test",
            "schema": 1,
            "method": "grouped_quadratic",
            "sequence_length": 4096,
        },
        data_identity={"manifest_sha256": "test-data"},
        model_identity={"identity_sha256": "test-model"},
    )
    parameters = {
        name: p for name, p in runtime.model.named_parameters() if p.requires_grad
    }
    identities = {name: id(p) for name, p in parameters.items()}
    for p in parameters.values():
        p.grad = torch.ones_like(p)
    runtime.optimizer.step()
    runtime.scheduler.step()
    runtime.optimizer.zero_grad(set_to_none=True)
    values = {name: p.detach().clone() for name, p in parameters.items()}
    moments = {
        name: copy.deepcopy(runtime.optimizer.state[p])
        for name, p in parameters.items()
    }
    scheduler_state = copy.deepcopy(runtime.scheduler.state_dict())
    cursor = {"next_train_block": 2, "steps_completed": 2}

    with preserve_global_rng_state():
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        save_latest_resume(
            tmp_path,
            model=runtime.model,
            kernel_bank=runtime.kernel_bank,
            context=context,
            step=2,
            validation_nll=1.0,
            optimizer=runtime.optimizer,
            scheduler=runtime.scheduler,
            cursor=cursor,
        )
        expected_random = (random.random(), np.random.random(), torch.rand(4))
        with torch.no_grad():
            for p in parameters.values():
                p.add_(0.5)
        runtime.optimizer.state.clear()
        runtime.scheduler.step()
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        restored = load_latest_resume(
            tmp_path,
            model=runtime.model,
            kernel_bank=runtime.kernel_bank,
            context=context,
            optimizer=runtime.optimizer,
            scheduler=runtime.scheduler,
        )
        actual_random = (random.random(), np.random.random(), torch.rand(4))

    assert restored.step == 2 and restored.cursor == cursor and restored.rng_restored
    assert runtime.scheduler.state_dict() == scheduler_state
    assert {name: id(p) for name, p in parameters.items()} == identities
    assert all(torch.equal(p, values[name]) for name, p in parameters.items())
    for name, p in parameters.items():
        assert runtime.optimizer.state[p].keys() == moments[name].keys()
        for key, value in runtime.optimizer.state[p].items():
            assert torch.equal(value, moments[name][key])
    assert actual_random[:2] == expected_random[:2]
    assert torch.equal(actual_random[2], expected_random[2])
