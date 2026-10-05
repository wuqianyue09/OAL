"""The released Qwen results must feed ordinary OAL training unchanged."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from qwen_lora_experiment.config import from_json_file
from qwen_lora_experiment.kernel_parameters import KernelParameterBank

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "configs" / "oal_qwen_n4096.json"


def _config():
    return replace(
        from_json_file(ROOT / "configs" / "lora_qwen_n4096_template.json"),
        group_asset_path=ASSET,
        group_asset_mode="precomputed",
        group_asset_sha256=None,
    )


def test_released_results_preserve_all_selected_groups_and_factor_values():
    from qwen_lora_experiment.precomputed_assets import load_precomputed_group_asset

    config = _config()
    config.validate()
    asset = load_precomputed_group_asset(config)
    bank = KernelParameterBank(
        "grouped_quadratic",
        grouped_initialization=asset.initialization,
        trainable_kernel_layer_ids=config.trainable_kernel_layer_ids,
    )
    source = json.loads(ASSET.read_text())
    assert len(asset.initialization.heads) == 252
    assert tuple(bank.dim_groups.shape) == (18, 14, 64)
    for (layer, head), record in asset.initialization.heads.items():
        old = source["layers"][str(layer)]["heads"][str(head)]
        assert record.groups == tuple((tuple(group) for group in old["groups"]))
        active_length = (record.group_count + 1) * (record.group_count + 2) // 2
        expected = torch.tensor(
            old["packed_lower_triangular"][:active_length], dtype=torch.float32
        )
        assert torch.equal(
            bank.active_grouped_factor_for_layer_head(layer, head).detach(), expected
        )


def test_precomputed_identity_records_source_without_inventing_calibration_provenance():
    from qwen_lora_experiment import runtime_identity
    from qwen_lora_experiment.paths import sha256_file

    config = _config()
    execution = {
        "method": config.method,
        "method_identity": config.method_identity,
        "execution": "triton",
        "callable": "oal_attention.oal_attention",
        "experimental": False,
        "admission": "public_fail_closed",
        "group_asset_mode": "precomputed",
        "group_asset_status": "precomputed",
        "group_asset_sequence_length": 4096,
        "replaced_layer_ids_zero_based": list(config.replacement_layer_ids),
    }
    identity = runtime_identity._experiment_identity(
        config,
        runtime=SimpleNamespace(
            kernel_bank=SimpleNamespace(
                trainable_kernel_layer_ids=config.trainable_kernel_layer_ids
            )
        ),
        attention_execution=execution,
        effective_config_sha256="a" * 64,
        data_manifest_sha256="b" * 64,
        model_identity={"model_name": "local-qwen"},
        data_manifest={},
        _formal_grouped_evidence_handoff=runtime_identity._FORMAL_GROUPED_EVIDENCE_HANDOFF,
    )
    assert identity["formal_admission"] == {
        "status": "precomputed_asset_loaded",
        "asset_status": "precomputed",
    }
    assert identity["adaptive_multigroup_asset"]["asset_sha256"] == sha256_file(ASSET)
    assert identity["adaptive_multigroup_asset"]["declared_sha256"] is None
    assert "provenance" not in identity["adaptive_multigroup_asset"]
    assert identity["factor_initialization_source"] == "precomputed_asset"


def test_precomputed_checkpoint_automatically_binds_the_asset_without_a_config_pin(
    tmp_path,
):
    from qwen_lora_experiment.checkpoint_identity import _checkpoint_config_identity
    from qwen_lora_experiment.paths import sha256_file

    config = _config()
    (tmp_path / "effective_config.json").write_text(json.dumps(config.to_dict()))
    identity = _checkpoint_config_identity(config, tmp_path, formal=False)
    assert identity["group_asset_sha256"] == sha256_file(ASSET)


def test_released_factors_and_lora_support_two_public_reference_backward_steps():
    from oal_attention import oal_attention
    from qwen_lora_experiment.lora import LoRAAdapter
    from qwen_lora_experiment.precomputed_assets import load_precomputed_group_asset

    config = replace(_config(), replacement_layers=(3,))
    bank = KernelParameterBank(
        "grouped_quadratic",
        grouped_initialization=load_precomputed_group_asset(config).initialization,
        trainable_kernel_layer_ids=(3,),
    )
    lora = LoRAAdapter(16, 896, rank=8, alpha=16, dropout=0, seed=42)
    generator = torch.Generator().manual_seed(73)
    inputs = torch.randn((1, 4, 16), generator=generator)
    base_q = torch.randn((1, 14, 4, 64), generator=generator)
    k = torch.randn((1, 2, 4, 64), generator=generator)
    v = torch.randn((1, 2, 4, 64), generator=generator)
    optimizer = torch.optim.SGD([*lora.parameters(), *bank.parameters()], lr=0.001)
    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        q = base_q + lora(inputs, output_dtype=torch.float32).reshape(
            1, 4, 14, 64
        ).transpose(1, 2)
        output = oal_attention(
            q,
            k,
            v,
            bank.dim_groups_for_layer(3),
            bank.parameters_for_layer(3),
            execution="reference",
        )
        loss = output.square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        for parameter in [lora.B, *bank.parameters(), *([lora.A] if step == 2 else [])]:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.norm() > 0
        optimizer.step()
