"""CPU-only contracts for explicit Qwen LoRA injection and ownership checks."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from qwen_lora_experiment.config import LoRAConfig
from qwen_lora_experiment.backbones.spec import ModelGeometry
from qwen_lora_experiment.lora import (
    ADAPTER_MODULE_NAME,
    LoRAAdapter,
    assert_all_on_device,
    assert_parameter_ownership,
    inject_lora_adapters,
    remove_lora_adapters,
    _expected_adapter_parameter_names,
    _expected_target_names,
)
from qwen_lora_experiment.protocol import resolve_seed_derivations

WIDTH = 4
TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")


class FakeSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for name in TARGET_NAMES:
            setattr(self, name, nn.Linear(WIDTH, WIDTH, bias=True))
        self.unrelated_projection = nn.Linear(WIDTH, WIDTH, bias=True)


class FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = FakeSelfAttention()
        self.mlp = nn.Linear(WIDTH, WIDTH, bias=False)


class FakeBackbone(nn.Module):
    def __init__(self, num_layers: int = 24) -> None:
        super().__init__()
        self.layers = nn.ModuleList(FakeLayer() for _ in range(num_layers))


class FakeQwen(nn.Module):
    def __init__(self, num_layers: int = 24) -> None:
        super().__init__()
        self.model = FakeBackbone(num_layers)
        self.lm_head = nn.Linear(WIDTH, WIDTH, bias=False)


class FakeAdapterWithLoRAParameterNames(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.ones(8, WIDTH, dtype=torch.float32))
        self.B = nn.Parameter(torch.ones(WIDTH, 8, dtype=torch.float32))


def make_model() -> FakeQwen:
    torch.manual_seed(11)
    return FakeQwen()


def expected_target_names() -> list[str]:
    return [
        f"model.layers.{layer}.self_attn.{projection}"
        for layer in range(24)
        for projection in TARGET_NAMES
    ]


def _geometry(num_layers: int) -> ModelGeometry:
    return ModelGeometry(
        num_layers=num_layers,
        hidden_size=WIDTH,
        num_query_heads=2,
        num_kv_heads=1,
        head_dim=2,
    )


def test_expected_names_and_injection_follow_each_instance_geometry() -> None:
    llama_geometry = _geometry(16)
    qwen_geometry = _geometry(24)

    assert len(_expected_target_names(llama_geometry)) == 64
    assert len(_expected_adapter_parameter_names(llama_geometry)) == 128

    observed_counts: list[int] = []
    for geometry in (qwen_geometry, llama_geometry, qwen_geometry):
        model = FakeQwen(geometry.num_layers)
        targets = inject_lora_adapters(
            model,
            LoRAConfig(),
            seed=42,
            geometry=geometry,
        )
        report = assert_parameter_ownership(
            model,
            method="softmax",
            geometry=geometry,
        )
        observed_counts.append(len(targets))
        assert len(report.trainable_names) == len(targets) * 2

    assert observed_counts == [96, 64, 96]


def test_injection_is_limited_to_96_exact_qwen_projection_paths() -> None:
    model = make_model()

    injected = inject_lora_adapters(model, LoRAConfig(), seed=42)

    assert injected == expected_target_names()
    assert len(injected) == 96
    for name in injected:
        linear = model.get_submodule(name)
        adapter = linear.get_submodule(ADAPTER_MODULE_NAME)
        assert isinstance(adapter, LoRAAdapter)
        assert adapter.rank == 8
        assert adapter.alpha == 16.0
        assert adapter.scale == 2.0
        assert linear.weight.requires_grad is False
        assert linear.bias is not None and linear.bias.requires_grad is False
        assert f"{name}.{ADAPTER_MODULE_NAME}.A" in model.state_dict()
        assert f"{name}.{ADAPTER_MODULE_NAME}.B" in model.state_dict()

    unrelated = model.model.layers[0].self_attn.unrelated_projection
    assert not hasattr(unrelated, ADAPTER_MODULE_NAME)
    assert unrelated.weight.requires_grad is False
    assert model.lm_head.weight.requires_grad is False
    assert {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    } == {
        f"{name}.{ADAPTER_MODULE_NAME}.{parameter_name}"
        for name in expected_target_names()
        for parameter_name in ("A", "B")
    }


def test_adapter_uses_seeded_fp32_kaiming_a_and_zero_b() -> None:
    adapter = LoRAAdapter(
        in_features=3,
        out_features=2,
        rank=2,
        alpha=4.0,
        dropout=0.0,
        seed=42,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    expected_a = torch.empty((2, 3), dtype=torch.float32)
    nn.init.kaiming_uniform_(expected_a, a=math.sqrt(5), generator=generator)

    assert adapter.A.is_leaf and adapter.B.is_leaf
    assert adapter.A.dtype is torch.float32
    assert adapter.B.dtype is torch.float32
    torch.testing.assert_close(adapter.A, expected_a, rtol=0, atol=0)
    assert torch.count_nonzero(adapter.B) == 0
    assert adapter.scale == 2.0


def test_injected_zero_b_preserves_the_exact_base_linear_output() -> None:
    model = make_model()
    target = model.model.layers[0].self_attn.q_proj
    inputs = torch.randn(2, 3, WIDTH)
    expected = target(inputs).detach().clone()

    inject_lora_adapters(model, LoRAConfig(), seed=42)

    assert torch.equal(target(inputs), expected)


def test_adapter_matches_standard_scale_and_training_dropout() -> None:
    adapter = LoRAAdapter(
        in_features=3,
        out_features=2,
        rank=2,
        alpha=6.0,
        dropout=0.5,
        seed=7,
    )
    with torch.no_grad():
        adapter.A.copy_(torch.tensor([[1.0, -2.0, 0.5], [0.25, 1.5, -1.0]]))
        adapter.B.copy_(torch.tensor([[1.0, 2.0], [-3.0, 0.5]]))
    inputs = torch.tensor([[1.0, 2.0, -1.0], [0.5, -0.25, 3.0]], dtype=torch.float64)
    adapter.train()

    torch.manual_seed(123)
    actual = adapter(inputs, output_dtype=torch.float64)
    torch.manual_seed(123)
    expected = F.linear(
        F.linear(F.dropout(inputs.float(), p=0.5, training=True), adapter.A), adapter.B
    )
    expected = expected * (6.0 / 2.0)

    assert actual.dtype is torch.float64
    torch.testing.assert_close(actual, expected.double(), rtol=0, atol=0)


def test_adapter_initialization_is_reproducible_for_the_same_seed() -> None:
    first = LoRAAdapter(4, 3, rank=2, alpha=4.0, dropout=0.0, seed=42)
    second = LoRAAdapter(4, 3, rank=2, alpha=4.0, dropout=0.0, seed=42)
    different = LoRAAdapter(4, 3, rank=2, alpha=4.0, dropout=0.0, seed=43)

    assert torch.equal(first.A, second.A)
    assert not torch.equal(first.A, different.A)


def test_formal_master_seed_controls_method_independent_lora_initialization() -> None:
    first = make_model()
    second = make_model()
    varied = make_model()
    shared_seed = resolve_seed_derivations(42).lora_init_seed

    inject_lora_adapters(first, LoRAConfig(), seed=shared_seed)
    inject_lora_adapters(second, LoRAConfig(), seed=shared_seed)
    inject_lora_adapters(
        varied,
        LoRAConfig(),
        seed=resolve_seed_derivations(73).lora_init_seed,
    )

    first_a = (
        first.get_submodule(expected_target_names()[0])
        .get_submodule(ADAPTER_MODULE_NAME)
        .A
    )
    second_a = (
        second.get_submodule(expected_target_names()[0])
        .get_submodule(ADAPTER_MODULE_NAME)
        .A
    )
    varied_a = (
        varied.get_submodule(expected_target_names()[0])
        .get_submodule(ADAPTER_MODULE_NAME)
        .A
    )
    assert torch.equal(first_a, second_a)
    assert not torch.equal(first_a, varied_a)


def test_repeated_injection_is_rejected() -> None:
    model = make_model()
    inject_lora_adapters(model, LoRAConfig(), seed=42)

    with pytest.raises(RuntimeError, match="already injected"):
        inject_lora_adapters(model, LoRAConfig(), seed=42)


def test_removal_removes_hooks_and_restores_base_output() -> None:
    model = make_model()
    target = model.model.layers[0].self_attn.q_proj
    inputs = torch.randn(2, WIDTH)
    expected = target(inputs).detach().clone()
    inject_lora_adapters(model, LoRAConfig(), seed=42)
    adapter = target.get_submodule(ADAPTER_MODULE_NAME)
    with torch.no_grad():
        adapter.B.fill_(1.0)

    removed = remove_lora_adapters(model)

    assert removed == expected_target_names()
    assert not hasattr(target, ADAPTER_MODULE_NAME)
    assert torch.equal(target(inputs), expected)


def test_removal_cleans_up_the_hook_when_its_adapter_was_externally_removed() -> None:
    model = make_model()
    target_name = expected_target_names()[0]
    target = model.get_submodule(target_name)
    inputs = torch.randn(2, WIDTH)
    expected = target(inputs).detach().clone()
    inject_lora_adapters(model, LoRAConfig(), seed=42)
    delattr(target, ADAPTER_MODULE_NAME)

    with pytest.raises(RuntimeError, match="missing its registered adapter"):
        target(inputs)

    removed = remove_lora_adapters(model)

    assert target_name in removed
    assert torch.equal(target(inputs), expected)


def test_removal_restores_original_base_trainability_and_clears_base_grads() -> None:
    model = make_model()
    model.model.layers[0].self_attn.q_proj.weight.requires_grad_(False)
    expected_trainability = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    inject_lora_adapters(model, LoRAConfig(), seed=42)
    remove_lora_adapters(model)

    assert {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    } == expected_trainability
    assert all(parameter.grad is None for parameter in model.parameters())


def test_parent_dtype_conversion_keeps_adapters_fp32_and_moves_them_with_the_model() -> (
    None
):
    model = make_model()
    inject_lora_adapters(model, LoRAConfig(), seed=42)

    model.to(dtype=torch.bfloat16)
    adapter = model.model.layers[0].self_attn.q_proj.get_submodule(ADAPTER_MODULE_NAME)

    assert adapter.A.dtype is torch.float32
    assert adapter.B.dtype is torch.float32
    assert model.model.layers[0].self_attn.q_proj.weight.dtype is torch.bfloat16
    assert (
        model.model.layers[0]
        .self_attn.q_proj(torch.ones(1, WIDTH, dtype=torch.bfloat16))
        .dtype
        is torch.bfloat16
    )

    model.to(device=torch.device("meta"))

    assert adapter.A.device.type == "meta"
    assert adapter.B.device.type == "meta"
    assert adapter.A.dtype is torch.float32
    assert adapter.B.dtype is torch.float32


def test_b_receives_a_gradient_before_a_and_a_receives_one_after_b_updates() -> None:
    adapter = LoRAAdapter(3, 2, rank=2, alpha=4.0, dropout=0.0, seed=42)
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
    inputs = torch.tensor([[1.0, -2.0, 3.0]])

    adapter(inputs, output_dtype=torch.float32).sum().backward()

    assert adapter.B.grad is not None and torch.isfinite(adapter.B.grad).all()
    assert torch.count_nonzero(adapter.B.grad) > 0
    assert adapter.A.grad is not None and torch.count_nonzero(adapter.A.grad) == 0
    optimizer.step()
    optimizer.zero_grad()
    assert torch.count_nonzero(adapter.B) > 0

    adapter(inputs, output_dtype=torch.float32).sum().backward()

    assert adapter.A.grad is not None and torch.isfinite(adapter.A.grad).all()
    assert torch.count_nonzero(adapter.A.grad) > 0


def test_ownership_report_allows_only_adapters_and_declared_kernel_parameters() -> None:
    model = make_model()
    inject_lora_adapters(model, LoRAConfig(), seed=42)
    model.kernel_factor = nn.Parameter(torch.ones(2))

    report = assert_parameter_ownership(
        model,
        method="grouped_quadratic",
        kernel_parameter_names=("kernel_factor",),
    )

    assert report.trainable_names == tuple(
        sorted(
            [
                *(
                    f"{name}.{ADAPTER_MODULE_NAME}.{parameter}"
                    for name in expected_target_names()
                    for parameter in ("A", "B")
                ),
                "kernel_factor",
            ]
        )
    )
    assert report.total_trainable_parameters == 96 * (WIDTH * 8 + 8 * WIDTH) + 2
    assert all(record.dtype is torch.float32 for record in report.trainable)

    model.lm_head.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="lm_head.weight"):
        assert_parameter_ownership(
            model,
            method="grouped_quadratic",
            kernel_parameter_names=("kernel_factor",),
        )


def test_ownership_audit_rejects_a_lookalike_adapter_module() -> None:
    model = make_model()
    inject_lora_adapters(model, LoRAConfig(), seed=42)
    target_name = expected_target_names()[0]
    target = model.get_submodule(target_name)
    delattr(target, ADAPTER_MODULE_NAME)
    target.add_module(ADAPTER_MODULE_NAME, FakeAdapterWithLoRAParameterNames())

    with pytest.raises(
        RuntimeError,
        match=rf"registered adapter must be LoRAAdapter: {target_name}",
    ):
        assert_parameter_ownership(model, method="softmax")


@pytest.mark.parametrize("missing_parameter", ("A", "B"))
def test_ownership_audit_requires_all_192_registered_adapter_parameters(
    missing_parameter: str,
) -> None:
    model = make_model()
    inject_lora_adapters(model, LoRAConfig(), seed=42)
    adapter = model.model.layers[0].self_attn.q_proj.get_submodule(ADAPTER_MODULE_NAME)
    delattr(adapter, missing_parameter)

    with pytest.raises(
        RuntimeError,
        match=rf"missing registered LoRA adapter parameter: .*\.{missing_parameter}",
    ):
        assert_parameter_ownership(model, method="softmax")


def test_injection_validates_the_lora_config_at_its_public_boundary() -> None:
    model = make_model()

    with pytest.raises(ValueError, match="lora.rank"):
        inject_lora_adapters(model, LoRAConfig(rank=4), seed=42)


def test_device_dtype_audit_names_the_offending_parameter() -> None:
    model = make_model()
    inject_lora_adapters(model, LoRAConfig(), seed=42)
    adapter = model.model.layers[0].self_attn.q_proj.get_submodule(ADAPTER_MODULE_NAME)
    adapter.A.data = adapter.A.data.double()

    with pytest.raises(
        RuntimeError,
        match=r"model\.layers\.0\.self_attn\.q_proj\._qwen_lora_adapter\.A.*device=cpu.*dtype=torch\.float64",
    ):
        assert_all_on_device(
            model, device=torch.device("cpu"), input_tensor=torch.ones(1, WIDTH)
        )
