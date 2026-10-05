"""Small CPU integration checks for the trained-adapter NLL -> PIQA workflow."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec
import json

import pytest
import torch
from torch import Tensor, nn

from qwen_lora_experiment.backbones.spec import ModelGeometry
from qwen_lora_experiment.checkpointing import CheckpointContext, save_best_adapter
from qwen_lora_experiment.config import from_json_file
from qwen_lora_experiment.grouped_initialization import (
    GroupedHeadParameters,
    GroupedParameterInitialization,
)
from qwen_lora_experiment.kernel_parameters import KernelParameterBank
from qwen_lora_experiment.lora import LoRAAdapter
from qwen_lora_experiment.telemetry import RunStateStore
from qwen_lora_experiment.workflows import evaluate

ROOT = Path(__file__).resolve().parents[1]


class TinyModel(nn.Module):
    """Four-token logits with real checkpointable LoRA and OAL parameters."""

    def __init__(self, kernel_bank: KernelParameterBank) -> None:
        super().__init__()
        self.adapter = LoRAAdapter(1, 4, rank=1, alpha=1.0, dropout=0.0, seed=0)
        self.kernel_bank = kernel_bank
        self.forward_calls = 0
        self.fail_forward = False

    def forward(self, *, input_ids: Tensor, use_cache: bool = False) -> object:
        assert use_cache is False
        self.forward_calls += 1
        if self.fail_forward:
            raise RuntimeError("tiny model forward failed")
        features = torch.ones((*input_ids.shape, 1), device=input_ids.device)
        residual = self.adapter(features, output_dtype=torch.float32)
        factor = self.kernel_bank.active_grouped_factor_for_layer_head(0, 0)[0]
        logits = residual + factor * torch.arange(4, device=input_ids.device)
        return SimpleNamespace(logits=logits)


class CharacterTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict:
        assert add_special_tokens is False
        return {"input_ids": [ord(character) % 4 for character in text]}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _persisted_status(run: SimpleNamespace) -> dict:
    return RunStateStore(run.prepared.run_dir / "status.json").read()


@pytest.fixture
def evaluation_run(tmp_path, monkeypatch):
    config = from_json_file(ROOT / "configs/lora_qwen_n4096_template.json")
    initialization = GroupedParameterInitialization(
        model_num_layers=1,
        num_query_heads=1,
        head_dim=64,
        epsilon=1e-6,
        layer_ids_zero_based=(0,),
        heads={
            (0, 0): GroupedHeadParameters(
                group_count=2,
                groups=(tuple(range(32)), tuple(range(32, 64))),
                packed_lower_triangular=(1.0, 0.0, 1.0, 0.0, 0.0, 1.0),
            )
        },
    )
    bank = KernelParameterBank(
        "grouped_quadratic",
        grouped_initialization=initialization,
        trainable_kernel_layer_ids=(0,),
        geometry=ModelGeometry(1, 64, 1, 1, 64),
    )
    model = TinyModel(bank)
    data_identity = {"manifest_sha256": "tiny-data"}
    model_identity = {"identity_sha256": "tiny-model"}
    context = CheckpointContext(
        method="grouped_quadratic",
        config_identity={
            "checkpoint_context_kind": "nonformal_test",
            "method": "grouped_quadratic",
            "sequence_length": config.sequence_length,
        },
        data_identity=data_identity,
        model_identity=model_identity,
    )
    store = RunStateStore(tmp_path / "status.json")
    store.create(run_id="tiny-run", method="grouped_quadratic")
    store.transition("running")
    store.transition("trained")
    save_best_adapter(
        tmp_path,
        model=model,
        kernel_bank=bank,
        context=context,
        step=2,
        validation_nll=1.0,
    )
    blocks = (torch.arange(config.sequence_length) % 4).unsqueeze(0)
    prepared = SimpleNamespace(
        run_dir=tmp_path,
        runtime=SimpleNamespace(
            model=model,
            kernel_bank=bank,
            tokenizer=CharacterTokenizer(),
            device="cpu",
        ),
        assets=SimpleNamespace(
            validation_mmap=blocks,
            test_mmap=blocks.flip(1),
            piqa_rows=(
                {
                    "id": "piqa:0",
                    "goal": "choose",
                    "sol1": "c",
                    "sol2": "a",
                    "label": 0,
                },
            ),
        ),
        checkpoint_context=context,
        execution={"attention": {"execution": "reference"}},
        data_identity=data_identity,
        model_identity=model_identity,
        source_identity={"source_files": []},
        experiment_identity={
            "experiment_protocol_version": "oal-qwen-lora-v1",
            "replaced_layer_ids_zero_based": [0],
            "method": "grouped_quadratic",
        },
        preloaded_piqa_rows=None,
        store=store,
    )
    # Replace model/data assembly only. Autospec enforces the real preparation
    # signature; workflow, checkpoint loading, scorers and publication stay real.
    preparation = create_autospec(
        evaluate._prepare_offline_evaluation, return_value=prepared
    )
    monkeypatch.setattr(evaluate, "_prepare_offline_evaluation", preparation)
    return SimpleNamespace(config=config, prepared=prepared, preparation=preparation)


def test_nll_then_piqa_restores_same_checkpoint_and_completes_run(evaluation_run):
    run = evaluation_run
    model = run.prepared.runtime.model
    saved = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    parameter_ids = {
        name: id(parameter) for name, parameter in model.named_parameters()
    }
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.5)

    nll = evaluate.execute_pilot_nll_evaluation(
        run.config,
        run_dir=run.prepared.run_dir,
        nonformal_test_mode=True,
    )
    assert nll.validation.token_count == run.config.sequence_length - 1
    assert nll.test.token_count == run.config.sequence_length - 1
    assert _persisted_status(run)["state"] == "trained"
    assert all(
        torch.equal(parameter, saved[name])
        for name, parameter in model.named_parameters()
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.5)
    piqa = evaluate.execute_pilot_piqa_evaluation(
        run.config,
        run_dir=run.prepared.run_dir,
        nonformal_test_mode=True,
    )
    nll_record = _read_json(nll.final_path)
    piqa_record = _read_json(piqa.final_path)
    assert piqa.best_checkpoint == nll.best_checkpoint == nll_record["best_checkpoint"]
    assert piqa_record["best_checkpoint"] == nll.best_checkpoint
    assert piqa.best_checkpoint["step"] == 2
    assert piqa.best_checkpoint["path"] == str(
        (run.prepared.run_dir / "best_adapter.pt").resolve()
    )
    assert piqa_record["pilot_nll_prerequisite"]["path"] == str(
        nll.final_path.resolve()
    )
    assert (
        nll_record["evaluation_scope"]
        == piqa_record["evaluation_scope"]
        == "pilot_single_seed"
    )
    assert piqa.piqa.question_count == 1
    assert piqa.piqa.raw_accuracy == piqa.piqa.acc_norm == 1.0
    predictions = [
        json.loads(line) for line in piqa.predictions_path.read_text().splitlines()
    ]
    assert predictions[0]["id"] == "piqa:0"
    assert predictions[0]["candidates"][0]["token_count"] == 2
    status = _persisted_status(run)
    assert status["state"] == "completed"
    assert status["nll_evaluation_path"] == str(nll.final_path.resolve())
    assert status["final_eval_path"] == str(piqa.final_path.resolve())
    assert model.forward_calls == 4
    assert model.training is True
    assert {
        name: id(parameter) for name, parameter in model.named_parameters()
    } == parameter_ids
    assert all(
        torch.equal(parameter, saved[name])
        for name, parameter in model.named_parameters()
    )
    assert all(
        call.kwargs["evaluation_mode"] == "pilot"
        for call in run.preparation.call_args_list
    )


def test_production_piqa_requires_this_runs_nll_before_model_preparation(
    evaluation_run,
):
    run = evaluation_run
    with pytest.raises(FileNotFoundError):
        evaluate.execute_pilot_piqa_evaluation(run.config, run_dir=run.prepared.run_dir)
    run.preparation.assert_not_called()
    assert run.prepared.runtime.model.forward_calls == 0
    assert _persisted_status(run)["state"] == "trained"
    assert not (run.prepared.run_dir / "pilot_piqa_eval.json").exists()


def test_piqa_rejects_checkpoint_replaced_after_nll_and_persists_failure(
    evaluation_run,
):
    run = evaluation_run
    nll = evaluate.execute_pilot_nll_evaluation(
        run.config,
        run_dir=run.prepared.run_dir,
        nonformal_test_mode=True,
    )
    original_record = nll.final_path.read_bytes()
    model = run.prepared.runtime.model
    with torch.no_grad():
        model.adapter.B.add_(0.25)
    updated = save_best_adapter(
        run.prepared.run_dir,
        model=model,
        kernel_bank=run.prepared.runtime.kernel_bank,
        context=run.prepared.checkpoint_context,
        step=3,
        validation_nll=0.5,
    )
    assert updated.saved
    calls_before_piqa = model.forward_calls
    with pytest.raises(
        ValueError, match="NLL and PIQA evaluation sources do not match"
    ):
        evaluate.execute_pilot_piqa_evaluation(
            run.config,
            run_dir=run.prepared.run_dir,
            nonformal_test_mode=True,
        )
    status = _persisted_status(run)
    assert status["state"] == "failed"
    assert status["failure"]["exception_type"] == "ValueError"
    assert (
        "NLL and PIQA evaluation sources do not match"
        in status["failure"]["exception_message"]
    )
    assert model.forward_calls == calls_before_piqa
    assert nll.final_path.read_bytes() == original_record
    assert not (run.prepared.run_dir / "pilot_piqa_eval.json").exists()
    assert not (
        run.prepared.run_dir / "pilot_piqa_validation_predictions.jsonl"
    ).exists()


@pytest.mark.parametrize("stage", ("nll", "piqa"))
def test_scoring_failure_is_durable_and_does_not_publish_endpoint(
    evaluation_run, stage
):
    run = evaluation_run
    nll_bytes = None
    if stage == "piqa":
        nll = evaluate.execute_pilot_nll_evaluation(
            run.config,
            run_dir=run.prepared.run_dir,
            nonformal_test_mode=True,
        )
        nll_bytes = nll.final_path.read_bytes()
    run.prepared.runtime.model.fail_forward = True
    runner = (
        evaluate.execute_pilot_nll_evaluation
        if stage == "nll"
        else evaluate.execute_pilot_piqa_evaluation
    )
    with pytest.raises(RuntimeError, match="tiny model forward failed"):
        runner(run.config, run_dir=run.prepared.run_dir, nonformal_test_mode=True)
    status = _persisted_status(run)
    assert status["state"] == "failed"
    assert status["failure"]["exception_type"] == "RuntimeError"
    assert status["failure"]["exception_message"] == "tiny model forward failed"
    assert not (run.prepared.run_dir / f"pilot_{stage}_eval.json").exists()
    assert not (
        run.prepared.run_dir / "pilot_piqa_validation_predictions.jsonl"
    ).exists()
    assert run.prepared.runtime.model.training is True
    if nll_bytes is not None:
        assert (run.prepared.run_dir / "pilot_nll_eval.json").read_bytes() == nll_bytes
