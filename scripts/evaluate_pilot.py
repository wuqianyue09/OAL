#!/usr/bin/env python3
"""Evaluate a trained OAL + LoRA checkpoint."""

from __future__ import annotations
import argparse
from collections.abc import Mapping
import importlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--stage",
        choices=("nll", "piqa", "mmlu", "hellaswag", "gsm8k", "arc_easy"),
        default="nll",
    )
    parser.add_argument("--evaluation-mode", choices=("pilot",), default="pilot")
    parser.add_argument("--operator-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--hd-manifest")
    parser.add_argument("--hd-probe")
    return parser


def main():
    args = build_parser().parse_args()
    from qwen_lora_experiment.workflows.config import load_pilot_config

    config = load_pilot_config(args.config)
    if (
        args.stage in {"hellaswag", "gsm8k", "arc_easy"}
        and Path(args.config).resolve()
        != (Path(args.run_dir) / "effective_config.json").resolve()
    ):
        raise ValueError(
            "use the trained run's effective_config.json for optional evaluation"
        )
    kwargs = {}
    if config.attention_backend == "hd_block_gemm":
        from qwen_lora_experiment.operator_imports import activate_operator_import_root

        activate_operator_import_root(args.operator_root)
        if args.hd_manifest is not None:
            kwargs["hd_manifest_path"] = args.hd_manifest
        if args.hd_probe is not None:
            kwargs["hd_probe_path"] = args.hd_probe
    modules = {
        "nll": "qwen_lora_experiment.workflows.evaluate",
        "piqa": "qwen_lora_experiment.workflows.evaluate",
        "mmlu": "qwen_lora_experiment.workflows.evaluate",
        "hellaswag": "qwen_lora_experiment.workflows.evaluate_hellaswag",
        "gsm8k": "qwen_lora_experiment.workflows.evaluate_gsm8k",
        "arc_easy": "qwen_lora_experiment.workflows.evaluate_arc_easy",
    }
    module = importlib.import_module(modules[args.stage])
    runner = getattr(module, f"execute_pilot_{args.stage}_evaluation")
    result = runner(config, run_dir=args.run_dir, evaluation_mode="pilot", **kwargs)
    print(json.dumps(_evaluation_result_record(result), sort_keys=True))
    return 0


def _evaluation_result_record(result: object) -> dict[str, object]:
    """Convert the production result dataclass into one explicit CLI record."""
    if isinstance(result, Mapping):
        return dict(result)
    if not hasattr(result, "final_path") or not hasattr(result, "best_checkpoint"):
        raise TypeError(
            "evaluation entrypoint must return a mapping or stage-result-like object"
        )
    best_checkpoint = getattr(result, "best_checkpoint")
    if not isinstance(best_checkpoint, Mapping):
        raise TypeError("evaluation result best_checkpoint must be a mapping")
    source = dict(best_checkpoint)
    record: dict[str, object] = {"final_path": str(getattr(result, "final_path"))}
    record["best_checkpoint"] = source
    if hasattr(result, "validation") and hasattr(result, "test"):
        validation_as_dict = getattr(getattr(result, "validation"), "as_dict", None)
        test_as_dict = getattr(getattr(result, "test"), "as_dict", None)
        if not callable(validation_as_dict) or not callable(test_as_dict):
            raise TypeError("NLL result scores must provide as_dict()")
        record.update(
            {"stage": "nll", "validation": validation_as_dict(), "test": test_as_dict()}
        )
        return record
    if hasattr(result, "predictions_path") and hasattr(result, "piqa"):
        piqa_summary = getattr(getattr(result, "piqa"), "summary", None)
        if not callable(piqa_summary):
            raise TypeError("PIQA result score must provide summary()")
        record.update(
            {
                "stage": "piqa",
                "predictions_path": str(getattr(result, "predictions_path")),
                "piqa": piqa_summary(),
            }
        )
        return record
    if hasattr(result, "predictions_path") and hasattr(result, "mmlu"):
        mmlu_summary = getattr(getattr(result, "mmlu"), "summary", None)
        if not callable(mmlu_summary):
            raise TypeError("MMLU result score must provide summary()")
        record.update(
            {
                "stage": "mmlu",
                "predictions_path": str(getattr(result, "predictions_path")),
                "mmlu": mmlu_summary(),
            }
        )
        return record
    for task in ("hellaswag", "gsm8k", "arc_easy"):
        if hasattr(result, "predictions_path") and hasattr(result, task):
            task_summary = getattr(getattr(result, task), "summary", None)
            if not callable(task_summary):
                raise TypeError(f"{task} result score must provide summary()")
            record.update(
                {
                    "stage": task,
                    "predictions_path": str(getattr(result, "predictions_path")),
                    task: task_summary(),
                }
            )
            return record
    raise TypeError(
        "evaluation result must be an NLL, PIQA, MMLU, HellaSwag, GSM8K, or ARC-E stage result"
    )


if __name__ == "__main__":
    raise SystemExit(main())
